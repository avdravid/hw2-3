#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define NUM_THREADS 256


double binSize;
int binNum;
particle_t* d_particles = nullptr;
particle_t* d_temp_particles = nullptr;
int* d_bin_counts = nullptr;
int* d_bin_starts = nullptr;
int* d_bin_offsets = nullptr;
void* d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff || r2 == 0) return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1.0 - cutoff / r) / r2 / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


__global__ void compute_bin_indices_kernel(particle_t* particles, int* counts, int n, double binSize, int binNum) {
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n; tid += blockDim.x * gridDim.x) {
        if (tid < n) {
            particle_t p = particles[tid];
            int bin_x = min(binNum - 1, max(0, int(p.x / binSize)));
            int bin_y = min(binNum - 1, max(0, int(p.y / binSize)));
            int bin_idx = bin_y * binNum + bin_x;
            atomicAdd(&counts[bin_idx], 1);
        }
    }
}

__global__ void sort_particles_kernel(particle_t* particles, particle_t* sorted, int* bin_starts, 
                                     int* bin_offsets, int n, double binSize, int binNum) {
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n; tid += blockDim.x * gridDim.x) {
        if (tid < n) {
            particle_t p = particles[tid];
            int bin_x = min(binNum - 1, max(0, int(p.x / binSize)));
            int bin_y = min(binNum - 1, max(0, int(p.y / binSize)));
            int bin_idx = bin_y * binNum + bin_x;
            int dest_idx = bin_starts[bin_idx] + atomicAdd(&bin_offsets[bin_idx], 1);
            sorted[dest_idx] = p;
        }
    }
}


__global__ void compute_forces_and_move_gpu(particle_t* binned_particles, particle_t* original_particles,
                                            int* bin_starts, int* bin_counts, int n, double binSize, 
                                            int binNum, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t* p = &original_particles[tid];
    p->ax = 0.0;
    p->ay = 0.0;

    int bin_x = min(binNum - 1, max(0, int(p->x / binSize)));
    int bin_y = min(binNum - 1, max(0, int(p->y / binSize)));


    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = bin_x + dx;
            int ny = bin_y + dy;
            if (nx < 0 || nx >= binNum || ny < 0 || ny >= binNum) continue;
            int bin_idx = ny * binNum + nx;
            int start = bin_starts[bin_idx];
            int count = bin_counts[bin_idx];
            for (int j = 0; j < count; j++) {
                particle_t neighbor = binned_particles[start + j];
                apply_force_gpu(*p, neighbor);
            }
        }
    }


    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;


    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}


void init_simulation(particle_t* parts_gpu, int num_parts, double size) {
    binSize = cutoff;
    binNum = max(1, int(size / binSize) + 1);

    d_particles = parts_gpu;
    CUDA_CHECK(cudaMalloc(&d_temp_particles, num_parts * sizeof(particle_t)));
    CUDA_CHECK(cudaMalloc(&d_bin_counts, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin_starts, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin_offsets, binNum * binNum * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_bin_counts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_starts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_offsets, 0, binNum * binNum * sizeof(int)));

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bin_counts, d_bin_starts, binNum * binNum);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}

void simulate_one_step(particle_t* parts_gpu, int num_parts, double size) {
    int blocks = min(108 * 4, (num_parts + NUM_THREADS - 1) / NUM_THREADS); 

    CUDA_CHECK(cudaMemset(d_bin_counts, 0, binNum * binNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bin_offsets, 0, binNum * binNum * sizeof(int)));

    compute_bin_indices_kernel<<<blocks, NUM_THREADS>>>(d_particles, d_bin_counts, num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bin_counts, d_bin_starts, binNum * binNum);
    CUDA_CHECK(cudaGetLastError());

    sort_particles_kernel<<<blocks, NUM_THREADS>>>(d_particles, d_temp_particles, d_bin_starts, d_bin_offsets, num_parts, binSize, binNum);
    CUDA_CHECK(cudaGetLastError());

    compute_forces_and_move_gpu<<<blocks, NUM_THREADS>>>(d_temp_particles, d_particles, d_bin_starts, d_bin_counts, num_parts, binSize, binNum, size);
    CUDA_CHECK(cudaGetLastError());
}

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
