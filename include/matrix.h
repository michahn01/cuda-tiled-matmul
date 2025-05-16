#pragma once

#include <vector>
#include <cstddef>

class Matrix {
public:

    /* -------------------------------------------------------------------- */
    /* -------------------------- CONSTRUCTORS ---------------------------- */
    /* -------------------------------------------------------------------- */

    /* Construct a 0x0 matrix with no data. */
    Matrix();

    /* Construct matrix with given dimensions, but all elements are uninitialized. */
    Matrix(size_t rows, size_t cols);

    /* Construct matrix with given dimensions; init all elements to specified value. */
    Matrix(size_t rows, size_t cols, float val);

    /* Construct a matrix from the given 2D vector. */
    Matrix(const std::vector<std::vector<float>>& mat);

    /* Construct an (N x 1) column vector from the given 1D vector. */
    Matrix(const std::vector<float>& col_vec);


    /* -------------------------------------------------------------------- */
    /* ---------------- MEMORY MANAGEMENT (rule of five) ------------------ */
    /* -------------------------------------------------------------------- */

    ~Matrix();
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;


    /* -------------------------------------------------------------------- */
    /* ---------------------------- UTILITY ------------------------------- */
    /* -------------------------------------------------------------------- */

    /* Note: [] accesses will not be bounds-checked. */
    float* operator[](size_t index);
    const float* operator[](size_t index) const;

    bool operator==(const Matrix& other);
    size_t getRows() const;
    size_t getCols() const;


    /* -------------------------------------------------------------------- */
    /* ---------------------- MATRIX MULTIPLICATION ----------------------- */
    /* -------------------------------------------------------------------- */

    /* Primary method for matrix multiplication; uses GPU and performs tiling. */
    Matrix operator*(const Matrix& other);

    enum class MatMulType {
        CPU, // utilize CPU for matmul
        GPU_NO_TILING, // utilize GPU for matmul, without tiling
        GPU_TILING // utilize GPU for matmul, with tiling; equivalent to invoking `operator*`
    };
    /* A matrix multiply method where caller can specify the type of implementation. */
    Matrix matmul(const Matrix& other, MatMulType type);


private:
    // Internally, data is represented as a flattened row-major array of floats.
    float* data = nullptr;
    size_t num_rows = 0;
    size_t num_cols = 0;
};
