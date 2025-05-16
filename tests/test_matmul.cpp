#include <cassert>
#include <cmath>

#include <vector>
#include <iostream>

#include "matrix.h"


/* -------------------------------------------------------------------- */
/* ------------------- MATRIX MULTIPLICATION TESTS -------------------- */
/* -------------------------------------------------------------------- */

void test_matmul_small() {
    std::cout << "Running `test_matmul_small`...\n";
    std::vector<std::vector<int>> A_data = {
        {-1, -1, -1},
        {2, 2, 2},
        {0, 0, 0},
        {1, 0, -1}
    };
    std::vector<std::vector<int>> B_data = {
        {1, 1},
        {0, 2},
        {0, -1},
    };
    Matrix A(A_data);
    Matrix B(B_data);

    Matrix C_cpu = A.matmul(B, Matrix::MatMulType::CPU);
    Matrix C_untiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_NO_TILING);
    Matrix C_tiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_TILING);
    
    assert(C_cpu == C_untiled_gpu);
    assert(C_untiled_gpu == C_tiled_gpu);
    assert(C_cpu == C_tiled_gpu);

    std::cout << "    > Passed `test_matmul_small`\n";
}

void test_matmul_medium() {
    std::cout << "Running `test_matmul_medium`...\n";

    int A_rows = 79;
    int width = 31;
    int B_cols = 33;

    std::vector<std::vector<int>> A_data;
    A_data.resize(A_rows);
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < width; ++j) {
            A_data[i].push_back((i + j) % 15);
        }
    }

    std::vector<std::vector<int>> B_data;
    B_data.resize(width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            B_data[i].push_back((i - 2 * j));
        }
    }

    Matrix A(A_data);
    Matrix B(B_data);

    Matrix C_cpu = A.matmul(B, Matrix::MatMulType::CPU);
    Matrix C_untiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_NO_TILING);
    Matrix C_tiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_TILING);
    
    assert(C_cpu == C_untiled_gpu);
    assert(C_untiled_gpu == C_tiled_gpu);
    assert(C_cpu == C_tiled_gpu);

    std::cout << "    > Passed `test_matmul_medium`\n";
}

void test_matmul_large() {
    std::cout << "Running `test_matmul_large`...\n";

    int A_rows = 643;
    int width = 499;
    int B_cols = 766;

    std::vector<std::vector<int>> A_data;
    A_data.resize(A_rows);
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < width; ++j) {
            A_data[i].push_back((i + j) % 29);
        }
    }

    std::vector<std::vector<int>> B_data;
    B_data.resize(width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            B_data[i].push_back((i - 2 * j));
        }
    }

    Matrix A(A_data);
    Matrix B(B_data);

    Matrix C_cpu = A.matmul(B, Matrix::MatMulType::CPU);
    Matrix C_untiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_NO_TILING);
    Matrix C_tiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_TILING);
    
    assert(C_cpu == C_untiled_gpu);
    assert(C_untiled_gpu == C_tiled_gpu);
    assert(C_cpu == C_tiled_gpu);

    std::cout << "    > Passed `test_matmul_large`\n";
}

void test_mat_vec_mult_large() {
    std::cout << "Running `test_mat_vec_mult_large`...\n";

    int A_rows = 643;
    int width = 499;

    std::vector<std::vector<int>> A_data;
    A_data.resize(A_rows);
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < width; ++j) {
            A_data[i].push_back(((i + j) % 54) - 11);
        }
    }

    std::vector<int> B_data;
    B_data.reserve(width);
    for (int i = 0; i < width; ++i) {
        int sign = (i % 2 == 0) ? 1 : -1;
        B_data.push_back(sign * (i * 2 - 43));
    }
    

    Matrix A(A_data);
    Matrix B(B_data); // initialize B as a (width x 1) column vector

    Matrix C_cpu = A.matmul(B, Matrix::MatMulType::CPU);
    Matrix C_untiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_NO_TILING);
    Matrix C_tiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_TILING);
    
    assert(C_cpu == C_untiled_gpu);
    assert(C_untiled_gpu == C_tiled_gpu);
    assert(C_cpu == C_tiled_gpu);

    std::cout << "    > Passed `test_mat_vec_mult_large`\n";
}

void test_consecutive_mults() {
    std::cout << "Running `test_consecutive_mults`...\n";

    int width = 66;

    std::vector<std::vector<int>> A_data;
    A_data.resize(width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            A_data[i].push_back(((i + j) % 6) - 11);
        }
    }
    Matrix A(A_data);

    std::vector<std::vector<int>> B_data;
    B_data.resize(width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            B_data[i].push_back((i - 2 * j) % 7);
        }
    }
    Matrix B(B_data);

    Matrix mat_cpu = A, mat_gpu_untiled = A, mat_gpu_tiled = A;

    // 4 consecutive Matrix-Matrix multiplications
    for (int i = 0; i < 4; ++i) {
        mat_cpu = B.matmul(mat_cpu, Matrix::MatMulType::CPU);
        mat_gpu_untiled = B.matmul(mat_gpu_untiled, Matrix::MatMulType::GPU_NO_TILING);
        mat_gpu_tiled = B.matmul(mat_gpu_tiled, Matrix::MatMulType::GPU_TILING);
    }

    assert(mat_cpu == mat_gpu_untiled);
    assert(mat_gpu_untiled == mat_gpu_tiled);
    assert(mat_cpu == mat_gpu_tiled);

    // Construct a vector
    std::vector<int> vec_data;
    vec_data.reserve(width);
    for (int i = 0; i < width; ++i) {
        int sign = (i % 2 == 0) ? 1 : -1;
        vec_data.push_back(sign * (i % 8));
    }
    Matrix vector(vec_data);

    // Multiply vector to accumulated matrices
    mat_cpu = mat_cpu.matmul(vector, Matrix::MatMulType::CPU);
    mat_gpu_untiled = mat_gpu_untiled.matmul(vector, Matrix::MatMulType::GPU_NO_TILING);
    mat_gpu_tiled = mat_gpu_tiled.matmul(vector, Matrix::MatMulType::GPU_TILING);

    assert(mat_cpu.getRows() == 66); assert(mat_cpu.getCols() == 1);
    assert(mat_gpu_untiled.getRows() == 66); assert(mat_gpu_untiled.getCols() == 1);
    assert(mat_gpu_tiled.getRows() == 66); assert(mat_gpu_tiled.getCols() == 1);

    assert(mat_cpu == mat_gpu_untiled);
    assert(mat_gpu_untiled == mat_gpu_tiled);
    assert(mat_cpu == mat_gpu_tiled);

    // Results should be same as if BBBBA * vec is done step-by-step
    Matrix vec_cpu = vector, vec_gpu_untiled = vector, vec_gpu_tiled = vector;

    vec_cpu = A.matmul(vec_cpu, Matrix::MatMulType::CPU);
    vec_gpu_untiled = A.matmul(vec_gpu_untiled, Matrix::MatMulType::GPU_NO_TILING);
    vec_gpu_tiled = A.matmul(vec_gpu_tiled, Matrix::MatMulType::GPU_TILING);

    for (int i = 0; i < 4; ++i) {
        vec_cpu = B.matmul(vec_cpu, Matrix::MatMulType::CPU);
        vec_gpu_untiled = B.matmul(vec_gpu_untiled, Matrix::MatMulType::GPU_NO_TILING);
        vec_gpu_tiled = B.matmul(vec_gpu_tiled, Matrix::MatMulType::GPU_TILING);
    }

    assert(vec_cpu == vec_gpu_untiled);
    assert(vec_gpu_untiled == vec_gpu_tiled);
    assert(vec_cpu == vec_gpu_tiled);

    assert(vec_cpu == mat_cpu);
    assert(vec_gpu_untiled == mat_gpu_untiled);
    assert(vec_gpu_tiled == mat_gpu_tiled);

    std::cout << "    > Passed `test_consecutive_mults`\n";
}


int main() {
    test_matmul_small();
    test_matmul_medium();
    test_matmul_large();
    test_mat_vec_mult_large();
    test_consecutive_mults();
}
