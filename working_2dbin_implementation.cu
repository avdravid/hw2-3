#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define NUM_THREADS 256

// Global binning parameters
double binSize;
int binNum;
particle_t* d_temp_particles = nullptr;
int* d_bin_counts = nullptr;

// Direction array for bin neighborhood lookup
__constant__ const int dir[9][2] = {
    {0, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, 
    {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
};

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

__global__ void get_count_kernel(particle_t* particles, int* counts, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    
    for (int i = tid; i < n; i += offset) {
        int x = int(particles[i].x / binSize);
        int y = int(particles[i].y / binSize);
        atomicAdd(counts + x * binNum + y, 1);
    }
}

__global__ void build_bins_kernel(particle_t* particles, particle_t* tmp, int* counts, 
                                int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    
    for (int i = tid; i < n; i += offset) {
        int x = int(particles[i].x / binSize);
        int y = int(particles[i].y / binSize);
        int id = atomicSub(counts + x * binNum + y, 1);
        tmp[id - 1] = particles[i];
    }
}

__global__ void compute_forces_gpu(particle_t* binned_particles, particle_t* original_particles, 
                                 int* counts, int n, double binSize, int binNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    
    for (int i = tid; i < n; i += offset) {
        // Get original particle to find bin location
        particle_t orig_p = original_particles[i];
        double ax = 0.0;
        double ay = 0.0;
        
        int bin_x = int(orig_p.x / binSize);
        int bin_y = int(orig_p.y / binSize);
        
        for (int t = 0; t < 9; t++) {
            int x = bin_x + dir[t][0];
            int y = bin_y + dir[t][1];
            
            if (x >= 0 && x < binNum && y >= 0 && y < binNum) {
                int id = x * binNum + y;
                int start = (id > 0) ? counts[id - 1] : 0;
                int end = counts[id];
                
                for (int j = start; j < end; j++) {
                    // We need to check the actual particle coordinates, not just indices
                    // because the binned particles are in a different order
                    particle_t& neighbor = binned_particles[j];
                    
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
        }
        
        // Store computed acceleration in the original particles array
        original_particles[i].ax = ax;
        original_particles[i].ay = ay;
    }
}

__global__ void move_gpu(particle_t* particles, int n, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    
    for (int i = tid; i < n; i += offset) {
        particle_t* p = &particles[i];
        
        // Velocity Verlet integration
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        
        // Bounce from walls
        while (p->x < 0 || p->x > size) {
            p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
            p->vx = -(p->vx);
        }
        while (p->y < 0 || p->y > size) {
            p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
            p->vy = -(p->vy);
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Initialize binning parameters
    binSize = cutoff * 2.0;
    binNum = int(size / binSize) + 1;
    
    // Allocate memory for temporary particles array and bin counts
    cudaMalloc(&d_temp_particles, num_parts * sizeof(particle_t));
    
    // Allocate bin counts with one extra element at the beginning
    cudaMalloc(&d_bin_counts, (binNum * binNum + 1) * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int blks = min(512, (num_parts + NUM_THREADS - 1) / NUM_THREADS);
    
    // Reset bin counts to zero
    cudaMemset(d_bin_counts, 0, (binNum * binNum + 1) * sizeof(int));
    
    // We'll use this pointer offset exactly like the reference code
    int* cnt = d_bin_counts + 1;  // Now cnt[-1] == 0
    
    // Count particles per bin
    get_count_kernel<<<blks, NUM_THREADS>>>(parts, cnt, num_parts, binSize, binNum);
    cudaDeviceSynchronize();
    
    // Copy bin counts to host and compute prefix sum
    int* host_counts = new int[binNum * binNum];
    cudaMemcpy(host_counts, cnt, binNum * binNum * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Prefix sum calculation
    for (int i = 1; i < binNum * binNum; i++) {
        host_counts[i] += host_counts[i - 1];
    }
    
    // Copy prefix sums back to device
    cudaMemcpy(cnt, host_counts, binNum * binNum * sizeof(int), cudaMemcpyHostToDevice);
    
    // Build binned structure
    build_bins_kernel<<<blks, NUM_THREADS>>>(parts, d_temp_particles, cnt, num_parts, binSize, binNum);
    cudaDeviceSynchronize();
    
    // Copy the original prefix sums for force computation
    cudaMemcpy(cnt, host_counts, binNum * binNum * sizeof(int), cudaMemcpyHostToDevice);
    
    // Compute forces using binned particles but update original particles
    // This is different from the reference code which swaps arrays
    compute_forces_gpu<<<blks, NUM_THREADS>>>(d_temp_particles, parts, cnt, num_parts, binSize, binNum);
    cudaDeviceSynchronize();
    
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
    
    // Clean up temporary host memory
    delete[] host_counts;
}
