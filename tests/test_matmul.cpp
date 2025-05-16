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

int main() {
    test_matmul_small();
    test_matmul_medium();
    test_matmul_large();
    test_mat_vec_mult_large();
}
