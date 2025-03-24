#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define NUM_THREADS 1024

// Global variables for binning and memory management
double binSize;                // Size of each bin, set to cutoff distance
int binNum;                    // Number of bins per dimension
particle_t* d_particles = nullptr;      // Device pointer to original particles
particle_t* d_temp_particles = nullptr; // Device pointer to sorted particles
int* d_bin_counts = nullptr;            // Number of particles per bin
int* d_bin_starts = nullptr;            // Starting index of each bin
int* d_bin_offsets = nullptr;           // Offsets for sorting particles into bins
void* d_temp_storage = nullptr;         // Temporary storage for CUB scan
size_t temp_storage_bytes = 0;          // Size of temporary storage

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device function to apply force between two particles
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Skip if distance exceeds cutoff or particles are the same
    if (r2 > cutoff * cutoff || r2 == 0) return;

    // Prevent division by very small distances
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    // Update acceleration
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Kernel to compute bin indices for each particle
__global__ void compute_bin_indices_kernel(particle_t* particles, int* counts, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t p = particles[tid];
    int bin_x = min(binNum - 1, max(0, int(p.x / binSize)));
    int bin_y = min(binNum - 1, max(0, int(p.y / binSize)));
    int bin_idx = bin_y * binNum + bin_x;  // Row-major order
    atomicAdd(&counts[bin_idx], 1);
}

// Kernel to sort particles into bins
__global__ void sort_particles_kernel(particle_t* particles, particle_t* sorted, int* bin_starts, 
                                     int* bin_offsets, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t p = particles[tid];
    int bin_x = min(binNum - 1, max(0, int(p.x / binSize)));
    int bin_y = min(binNum - 1, max(0, int(p.y / binSize)));
    int bin_idx = bin_y * binNum + bin_x;  // Row-major order
    int dest_idx = bin_starts[bin_idx] + atomicAdd(&bin_offsets[bin_idx], 1);
    sorted[dest_idx] = p;
}

// Kernel to compute forces using binned particles
__global__ void compute_forces_gpu(particle_t* binned_particles, particle_t* original_particles,
                                   int* bin_starts, int* bin_counts, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t* p = &original_particles[tid];
    p->ax = 0.0;
    p->ay = 0.0;
    int bin_x = min(binNum - 1, max(0, int(p->x / binSize)));
    int bin_y = min(binNum - 1, max(0, int(p->y / binSize)));

    // Check neighboring bins (3x3 grid)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = bin_x + dx;
            int ny = bin_y + dy;
            if (nx < 0 || nx >= binNum || ny < 0 || ny >= binNum) continue;

            int bin_idx = ny * binNum + nx;  // Row-major order
            int start = bin_starts[bin_idx];
            int count = bin_counts[bin_idx];

            // Apply forces from particles in this bin
            for (int j = 0; j < count; j++) {
                particle_t neighbor = binned_particles[start + j];
                apply_force_gpu(*p, neighbor);
            }
        }
    }
}

// Kernel to update particle positions and velocities
__global__ void move_gpu(particle_t* particles, int n, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t* p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // Handle periodic boundary conditions
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

// Initialize the simulation
void init_simulation(particle_t* parts_gpu, int num_parts, double size) {
    binSize = cutoff;  // Bin size matches cutoff for efficient force calculation
    binNum = max(1, int(size / binSize) + 1);

    d_particles = parts_gpu;
    CUDA_CHECK(cudaMalloc(&d_temp_particles, num_parts * sizeof(particle_t)));
    CUDA_CHECK(cudaMalloc(&d_bin_counts, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin_starts, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin_offsets, binNum * binNum * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_bin_counts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_starts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_offsets, 0, binNum * binNum * sizeof(int)));

    // Allocate temporary storage for CUB prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bin_counts, d_bin_starts, binNum * binNum);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}

// Simulate one time step
void simulate_one_step(particle_t* parts_gpu, int num_parts, double size) {
    int blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Reset binning data
    CUDA_CHECK(cudaMemset(d_bin_counts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_offsets, 0, binNum * binNum * sizeof(int)));

    // Compute bin indices
    compute_bin_indices_kernel<<<blocks, NUM_THREADS>>>(d_particles, d_bin_counts, num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());

    // Compute bin starting indices using CUB
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bin_counts, d_bin_starts, binNum * binNum);
    CUDA_CHECK(cudaGetLastError());

    // Sort particles into bins
    sort_particles_kernel<<<blocks, NUM_THREADS>>>(d_particles, d_temp_particles, d_bin_starts, d_bin_offsets, num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());

    // Compute forces
    compute_forces_gpu<<<blocks, NUM_THREADS>>>(d_temp_particles, d_particles, d_bin_starts, d_bin_counts, num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());

    // Update particle positions
    move_gpu<<<blocks, NUM_THREADS>>>(d_particles, num_parts, size);
    CUDA_CHECK(cudaGetLastError());
}

// Free allocated resources
void free_simulation_resources() {
    if (d_temp_particles) cudaFree(d_temp_particles);
    if (d_bin_counts) cudaFree(d_bin_counts);
    if (d_bin_starts) cudaFree(d_bin_starts);
    if (d_bin_offsets) cudaFree(d_bin_offsets);
    if (d_temp_storage) cudaFree(d_temp_storage);

    d_temp_particles = nullptr;
    d_bin_counts = nullptr;
    d_bin_starts = nullptr;
    d_bin_offsets = nullptr;
    d_temp_storage = nullptr;
}
