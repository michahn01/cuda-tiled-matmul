#include <stdio.h>
#include <cmath>
#include <cstring>
#include <cassert>

#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>

#include "matrix.h"

#define TILE_WIDTH 16
#define EPSILON 0.00001

#define CUDA_ASSERT(res) ( check_cuda_error((res), __FILE__, __LINE__) )
inline void check_cuda_error(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA check failed in file %s line %d: %s", file, line, cudaGetErrorString(code));
        exit(code);
    }
}


/* -------------------------------------------------------------------- */
/* -------------------- MATMUL: CPU IMPLEMENTATION -------------------- */
/* -------------------------------------------------------------------- */

/*
Use the CPU to perform matrix multiplication on pointers to 1D row-major
matrix representations of floats.

Assumes dimensions match and `output` points to well-defined memory.
*/
void cpu_matmul(
    float* A, float* B, float* output,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float entry = 0;
            for (int k = 0; k < A_cols; ++k) {
                entry += A[i * A_cols + k] * B[k * B_cols + j];
            }
            output[i * B_cols + j] = entry;
        }
    }
}


/* -------------------------------------------------------------------- */
/* ---------------- MATMUL: UNTILED GPU IMPLEMENTATION ---------------- */
/* -------------------------------------------------------------------- */

/* Note: TILED version of GPU matmul is also implemented further down this file. */

__global__
void untiled_gpu_matmul_kernel(
    float* A, float* B, float* output,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float entry = 0;
        for (int k = 0; k < A_cols; ++k) {
            entry += A[row * A_cols + k] * B[k * B_cols + col];
        }
        output[row * B_cols + col] = entry;
    }
}

/*
Use the GPU to perform matrix multiplication on pointers to 1D row-major
matrix representations of floats. Does not utilize tiling. 

Assumes dimensions match and `output` points to well-defined memory.
*/
void untiled_gpu_matmul(
    float* A_h, float* B_h, float* output_h,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    uint32_t A_size = A_rows * A_cols * sizeof(float);
    uint32_t B_size = B_rows * B_cols * sizeof(float);
    uint32_t output_size = A_rows * B_cols * sizeof(float);

    // Allocate memory on device (GPU).
    float* A_d;
    float* B_d;
    float* output_d;
    cudaMalloc((void**)&A_d, A_size);
    cudaMalloc((void**)&B_d, B_size);
    cudaMalloc((void**)&output_d, output_size);

    // Copy matrices from host to device.
    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

    // Configure dimensions and launch kernel.
    dim3 block_dim(32, 32);
    dim3 grid_dim(ceil(B_cols / 32.0), ceil(A_rows / 32.0));
    untiled_gpu_matmul_kernel<<<grid_dim, block_dim>>>(
        A_d, B_d, output_d,
        A_rows, A_cols, B_rows, B_cols
    );

    // Check that kernel launch was successful.
    GPU_ASSERT(cudaGetLastError());

    // Copy answer from device to host.
    GPU_ASSERT(cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost));

    // Free memory and return.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(output_d);
}


/* -------------------------------------------------------------------- */
/* ----------------- MATMUL: TILED GPU IMPLEMENTATION ----------------- */
/* -------------------------------------------------------------------- */

__global__
void gpu_matmul_kernel(
    float* A, float* B, float* output,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    extern __shared__ float Ads[];
    float* Bds = Ads + TILE_WIDTH * TILE_WIDTH;

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int tile_idx = ty * TILE_WIDTH + tx;

    uint32_t row = by * blockDim.y + ty;
    uint32_t col = bx * blockDim.x + tx;

    float entry = 0;

    // Repeatedly process tiles across A_cols (equal to B_rows).
    for (int tile = 0; tile < ceil(float(A_cols) / TILE_WIDTH); ++tile) {
        // (PHASE 1) Collaboratively load data into shared memory pool.
        if (row < A_rows && tile * TILE_WIDTH + tx < A_cols) {
            Ads[tile_idx] = A[row * A_cols + tile * TILE_WIDTH + tx];
        }
        else {
            Ads[tile_idx] = 0;
        }
        if (col < B_cols && tile * TILE_WIDTH + ty < B_rows) {
            Bds[tile_idx] = B[col + (tile * TILE_WIDTH + ty) * B_cols];
        }
        else {
            Bds[tile_idx] = 0;
        }

        // Wait for all threads to finish filling up shared memory.
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            // Ads[ty][k] * Bds[k][tx]
            entry += (
                Ads[ty * TILE_WIDTH + k] *
                Bds[k * TILE_WIDTH + tx]
            );
        }

        // Wait for all threads to finish reading this round of shared memory.
        __syncthreads();
    }

    if (row < A_rows && col < B_cols) {
        output[row * B_cols + col] = entry;
    }
}

/*
Use the GPU to perform matrix multiplication on pointers to 1D row-major
matrix representations of floats. Utilizes tiling.

Assumes dimensions match and `output` points to well-defined memory.
*/
void gpu_matmul(
    float* A_h, float* B_h, float* output_h,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {

    uint32_t A_size = A_rows * A_cols * sizeof(float);
    uint32_t B_size = B_rows * B_cols * sizeof(float);
    uint32_t output_size = A_rows * B_cols * sizeof(float);

    // Allocate memory on device (GPU).
    float* A_d;
    float* B_d;
    float* output_d;
    cudaMalloc((void**)&A_d, A_size);
    cudaMalloc((void**)&B_d, B_size);
    cudaMalloc((void**)&output_d, output_size);

    // Copy matrices from host to device.
    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

    // Configure dimensions and launch kernel.
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        ceil(double(B_cols) / TILE_WIDTH),
        ceil(double(A_rows) / TILE_WIDTH)
    );

    int shared_mem_bytes = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    gpu_matmul_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
        A_d, B_d, output_d,
        A_rows, A_cols, B_rows, B_cols
    );

    // Check that kernel launch was successful.
    GPU_ASSERT(cudaGetLastError());

    // Copy answer from device to host.
    GPU_ASSERT(cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost));

    // Free memory and return.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(output_d);
}
