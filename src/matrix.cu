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
