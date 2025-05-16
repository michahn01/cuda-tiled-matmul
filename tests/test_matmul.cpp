#include <cassert>
#include <vector>

#include "matrix.h"

void test_matmul_small() {
    std::vector<std::vector<float>> A = {
        {-1, -1, -1},
        {2, 2, 2},
        {0, 0, 0},
        {1, 0, -1}
    };
    std::vector<std::vector<float>> B = {
        {1, 1},
        {0, 2},
        {0, -1},
    };

    Matrix C_cpu = A.matmul(B, Matrix::MatMulType::CPU);
    Matrix C_untiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_NO_TILING);
    Matrix C_tiled_gpu = A.matmul(B, Matrix::MatMulType::GPU_TILING);
    
    assert(C_cpu == C_untiled_gpu);
    assert(C_untiled_gpu == C_tiled_gpu);
    assert(C_cpu == C_tiled_gpu);
}

int main() {
    test_matmul_small();
}
