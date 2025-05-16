#include <stdio.h>
#include <cmath>
#include <cstring>
#include <cassert>

#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>

#include "matrix.h"

#define TILE_WIDTH 16

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
matrix representations of ints.

Assumes dimensions match and `output` points to well-defined memory.
*/
void cpu_matmul(
    int* A, int* B, int* output,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            int entry = 0;
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
    int* A, int* B, int* output,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        int entry = 0;
        for (int k = 0; k < A_cols; ++k) {
            entry += A[row * A_cols + k] * B[k * B_cols + col];
        }
        output[row * B_cols + col] = entry;
    }
}

/*
Use the GPU to perform matrix multiplication on pointers to 1D row-major
matrix representations of ints. Does not utilize tiling. 

Assumes dimensions match and `output` points to well-defined memory.
*/
void untiled_gpu_matmul(
    int* A_h, int* B_h, int* output_h,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    uint32_t A_size = A_rows * A_cols * sizeof(int);
    uint32_t B_size = B_rows * B_cols * sizeof(int);
    uint32_t output_size = A_rows * B_cols * sizeof(int);

    // Allocate memory on device (GPU).
    int* A_d;
    int* B_d;
    int* output_d;
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
    CUDA_ASSERT(cudaGetLastError());

    // Copy answer from device to host.
    CUDA_ASSERT(cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost));

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
    int* A, int* B, int* output,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {
    extern __shared__ int Ads[];
    int* Bds = Ads + TILE_WIDTH * TILE_WIDTH;

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int tile_idx = ty * TILE_WIDTH + tx;

    uint32_t row = by * blockDim.y + ty;
    uint32_t col = bx * blockDim.x + tx;

    int entry = 0;

    // Repeatedly process tiles across A_cols (equal to B_rows).
    for (int tile = 0; tile < ceil(double(A_cols) / TILE_WIDTH); ++tile) {
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
matrix representations of ints. Utilizes tiling.

Assumes dimensions match and `output` points to well-defined memory.
*/
void gpu_matmul(
    int* A_h, int* B_h, int* output_h,
    uint32_t A_rows, uint32_t A_cols,
    uint32_t B_rows, uint32_t B_cols
) {

    uint32_t A_size = A_rows * A_cols * sizeof(int);
    uint32_t B_size = B_rows * B_cols * sizeof(int);
    uint32_t output_size = A_rows * B_cols * sizeof(int);

    // Allocate memory on device (GPU).
    int* A_d;
    int* B_d;
    int* output_d;
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

    int shared_mem_bytes = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(int);
    gpu_matmul_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
        A_d, B_d, output_d,
        A_rows, A_cols, B_rows, B_cols
    );

    // Check that kernel launch was successful.
    CUDA_ASSERT(cudaGetLastError());

    // Copy answer from device to host.
    CUDA_ASSERT(cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost));

    // Free memory and return.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(output_d);
}


/* -------------------------------------------------------------------- */
/* ------------------ `MATRIX` CLASS IMPLEMENTATION ------------------- */
/* -------------------------------------------------------------------- */


/* --- CONSTRUCTORS --- */

/* Construct a 0x0 matrix with no data. */
Matrix::Matrix() {}

/* Construct matrix with given dimensions, but all elements are uninitialized. */
Matrix::Matrix(size_t rows, size_t cols) :
    data(new int[rows * cols]), num_rows(rows), num_cols(cols) {}

/* Construct matrix with given dimensions; init all elements to specified value. */
Matrix::Matrix(size_t rows, size_t cols, int val) :
    data(new int[rows * cols]), num_rows(rows), num_cols(cols) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = val;
    }
}

/* Construct a matrix from the given 2D vector. */
Matrix::Matrix(const std::vector<std::vector<int>>& mat) :
    num_rows(mat.size()), num_cols(mat.empty() ? 0 : mat[0].size()) {

    data = new int[num_rows * num_cols];
    int* data_row = data;

    for (const auto& row : mat) {
        if (row.size() != num_cols) {
            delete[] data;
            throw std::runtime_error("All rows of matrix must be equal length.");
        }
        memcpy(data_row, row.data(), num_cols * sizeof(int));
        data_row += num_cols;
    }
}

/* Construct an (N x 1) column vector from the given 1D vector. */
Matrix::Matrix(const std::vector<int>& col_vec) {
    num_rows = col_vec.size();
    num_cols = 1;
    data = new int[num_rows * num_cols];
    memcpy(data, col_vec.data(), num_rows * num_cols * sizeof(int));
}


/* --- MEMORY MANAGEMENT (rule of five) --- */

Matrix::~Matrix() {
    delete[] data;
}

Matrix::Matrix(const Matrix& other) {
    num_rows = other.num_rows;
    num_cols = other.num_cols;
    
    size_t mat_len = num_rows * num_cols;
    data = new int[mat_len];

    if (mat_len != 0 && other.data == nullptr) {
        throw std::runtime_error("Source data is nullptr when non-empty data expected.");
    }
    memcpy(data, other.data, mat_len * sizeof(int));
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] data;
        num_rows = other.num_rows;
        num_cols = other.num_cols;
        size_t mat_len = num_rows * num_cols;
        data = new int[mat_len];
        memcpy(data, other.data, mat_len * sizeof(int));
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : data(other.data), num_rows(other.num_rows), num_cols(other.num_cols) {
    other.data = nullptr;
    other.num_rows = 0;
    other.num_cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data;

        data = other.data;
        num_rows = other.num_rows;
        num_cols = other.num_cols;

        other.data = nullptr;
        other.num_rows = 0;
        other.num_cols = 0;
    }
    return *this;
}


/* --- UTILITY --- */

int* Matrix::operator[](size_t index) {
    return data + (index * num_cols);
}
const int* Matrix::operator[](size_t index) const {
    return data + (index * num_cols);
}
bool Matrix::operator==(const Matrix& other) {
    if (num_rows != other.num_rows || num_cols != other.num_cols) {
        return false;
    }

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            if ((*this)[i][j] != other[i][j]) {
                return false;
            }
        }
    }

    return true;
}
size_t Matrix::getRows() const {
    return num_rows;
}
size_t Matrix::getCols() const {
    return num_cols;
}


/* --- MATRIX MULTIPLICATION --- */

Matrix Matrix::operator*(const Matrix& other) {
    if (num_cols != other.num_rows) {
        throw std::runtime_error("Matrices must have compatible dimensions.");
    }

    Matrix result(num_rows, other.num_cols);
    gpu_matmul(
        data, other.data, result.data,
        num_rows, num_cols,
        other.num_rows, other.num_cols
    );
    return result;
}

Matrix Matrix::matmul(const Matrix& other, MatMulType type) {
    if (num_cols != other.num_rows) {
        throw std::runtime_error("Matrices must have compatible dimensions.");
    }

    Matrix result(num_rows, other.num_cols);

    if (type == Matrix::MatMulType::CPU) {
        cpu_matmul(
            data, other.data, result.data,
            num_rows, num_cols,
            other.num_rows, other.num_cols
        );
    }
    else if (type == Matrix::MatMulType::GPU_NO_TILING) {
        untiled_gpu_matmul(
            data, other.data, result.data,
            num_rows, num_cols,
            other.num_rows, other.num_cols
        );
    }
    else if (type == Matrix::MatMulType::GPU_TILING) {
        gpu_matmul(
            data, other.data, result.data,
            num_rows, num_cols,
            other.num_rows, other.num_cols
        );
    }
    return result;
}
