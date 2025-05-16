set(
    MATRIX_SOURCES
    matrix.cu
)

# Personal note: if on NVIDIA T4, don't forget to
# configure CMake with: `-DCMAKE_CUDA_ARCHITECTURES=75`
add_library(matrix STATIC
    ${MATRIX_SOURCES}
)
