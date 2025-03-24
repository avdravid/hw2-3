#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cub/cub.cuh>  // CUB library for efficient parallel primitives

#define NUM_THREADS 256

// Global binning parameters
double binSize;
int binNum;
particle_t* d_particles = nullptr;       // Main particle array on device
particle_t* d_temp_particles = nullptr;  // Temporary particles array for binning
int* d_bin_counts = nullptr;             // Bin counts
int* d_bin_starts = nullptr;             // Starting indices for each bin
void* d_temp_storage = nullptr;          // Temporary storage for CUB operations
size_t temp_storage_bytes = 0;           // Size of temporary storage

// Direction array for bin neighborhood lookup
__constant__ const int dir[9][2] = {
    {0, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0},
    {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
};

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    // Simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_bin_indices_kernel(particle_t* particles, int* counts, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t p = particles[tid];
    int bin_x = min(binNum-1, max(0, int(p.x / binSize)));
    int bin_y = min(binNum-1, max(0, int(p.y / binSize)));
    int bin_idx = bin_x * binNum + bin_y;
    
    atomicAdd(&counts[bin_idx], 1);
}

__global__ void sort_particles_kernel(particle_t* particles, particle_t* sorted, int* bin_starts, 
                                      int* bin_offsets, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t p = particles[tid];
    int bin_x = min(binNum-1, max(0, int(p.x / binSize)));
    int bin_y = min(binNum-1, max(0, int(p.y / binSize)));
    int bin_idx = bin_x * binNum + bin_y;
    
    int dest_idx = bin_starts[bin_idx] + atomicAdd(&bin_offsets[bin_idx], 1);
    sorted[dest_idx] = p;
}

__global__ void compute_forces_gpu(particle_t* binned_particles, particle_t* original_particles,
                                  int* bin_starts, int* bin_counts,
                                  int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Get original particle to find bin location
    particle_t orig_p = original_particles[tid];
    double ax = 0.0;
    double ay = 0.0;

    int bin_x = min(binNum-1, max(0, int(orig_p.x / binSize)));
    int bin_y = min(binNum-1, max(0, int(orig_p.y / binSize)));

    // Loop through neighboring bins
    for (int i = 0; i < 9; i++) {
        int nx = bin_x + dir[i][0];
        int ny = bin_y + dir[i][1];
        
        if (nx < 0 || nx >= binNum || ny < 0 || ny >= binNum)
            continue;
            
        int bin_idx = nx * binNum + ny;
        int start = bin_starts[bin_idx];
        int count = bin_counts[bin_idx];
        
        // Process each particle in this bin
        for (int j = 0; j < count; j++) {
            particle_t& neighbor = binned_particles[start + j];
            
            // Check if this is the same particle (approximately)
            if (fabs(neighbor.x - orig_p.x) < 1e-10 &&
                fabs(neighbor.y - orig_p.y) < 1e-10)
                continue;
            
            double dx = neighbor.x - orig_p.x;
            double dy = neighbor.y - orig_p.y;
            double r2 = dx * dx + dy * dy;

            if (r2 > cutoff * cutoff)
                continue;

            r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
            double r = sqrt(r2);

            double coef = (1 - cutoff / r) / r2 / mass;
            ax += coef * dx;
            ay += coef * dy;
        }
    }

    // Store computed acceleration in the original particles array
    original_particles[tid].ax = ax;
    original_particles[tid].ay = ay;
}

__global__ void move_gpu(particle_t* particles, int n, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t* p = &particles[tid];

    // Velocity Verlet integration
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // Bounce from walls - using while loops for correctness
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Initialize binning parameters
    binSize = cutoff * 2.0;
    binNum = max(1, int(size / binSize) + 1);
    
    // Allocate device memory - keep particles on GPU throughout simulation
    CUDA_CHECK(cudaMalloc(&d_particles, num_parts * sizeof(particle_t)));
    CUDA_CHECK(cudaMalloc(&d_temp_particles, num_parts * sizeof(particle_t)));
    CUDA_CHECK(cudaMalloc(&d_bin_counts, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin_starts, binNum * binNum * sizeof(int)));
    
    // Copy initial particle data to device
    CUDA_CHECK(cudaMemcpy(d_particles, parts, num_parts * sizeof(particle_t), cudaMemcpyHostToDevice));
    
    // Ensure all GPU memory is initialized properly
    CUDA_CHECK(cudaMemset(d_bin_counts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_starts, 0, binNum * binNum * sizeof(int)));
    
    // Get required temporary storage size for CUB
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                  d_bin_counts, d_bin_starts, binNum * binNum);
    
    // Allocate temporary storage for CUB
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Warm up GPU to avoid initial performance hit
    cudaDeviceSynchronize();
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    
    // Reset bin counts
    CUDA_CHECK(cudaMemset(d_bin_counts, 0, binNum * binNum * sizeof(int)));
    
    // Step 1: Count particles per bin
    compute_bin_indices_kernel<<<blocks, NUM_THREADS>>>(
        d_particles, d_bin_counts, num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Compute bin start indices (prefix sum) using CUB
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                 d_bin_counts, d_bin_starts, binNum * binNum);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 3: Create temp array with bin offsets for sorting
    int* d_bin_offsets;
    CUDA_CHECK(cudaMalloc(&d_bin_offsets, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_offsets, 0, binNum * binNum * sizeof(int)));
    
    // Step 4: Sort particles into bins
    sort_particles_kernel<<<blocks, NUM_THREADS>>>(
        d_particles, d_temp_particles, d_bin_starts, d_bin_offsets, 
        num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 5: Compute forces - maintaining the approach from the first code
    // where we compute forces on the original particles using binned particles
    compute_forces_gpu<<<blocks, NUM_THREADS>>>(
        d_temp_particles, d_particles, d_bin_starts, d_bin_counts, 
        num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 6: Move particles
    move_gpu<<<blocks, NUM_THREADS>>>(d_particles, num_parts, size);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(parts, d_particles, num_parts * sizeof(particle_t), 
                         cudaMemcpyDeviceToHost));
    
    // Free temporary memory
    CUDA_CHECK(cudaFree(d_bin_offsets));
}

void free_simulation_resources() {
    if (d_particles) cudaFree(d_particles);
    if (d_temp_particles) cudaFree(d_temp_particles);
    if (d_bin_counts) cudaFree(d_bin_counts);
    if (d_bin_starts) cudaFree(d_bin_starts);
    if (d_temp_storage) cudaFree(d_temp_storage);
    
    d_particles = nullptr;
    d_temp_particles = nullptr;
    d_bin_counts = nullptr;
    d_bin_starts = nullptr;
    d_temp_storage = nullptr;
}
