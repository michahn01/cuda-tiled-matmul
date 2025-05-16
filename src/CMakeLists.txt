set(
    MATRIX_SOURCES
    matrix.cu
)

add_library(matrix STATIC
    ${MATRIX_SOURCES}
)
