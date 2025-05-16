cmake_minimum_required(VERSION 3.18)
project(cuda_matmul LANGUAGES CXX CUDA)

# Specify C++ version for host C++ compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Specify C++ version for NVCC
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)


# Personal reminder:
# - If on T4 GPU, configure CMake with: `-DCMAKE_CUDA_ARCHITECTURES=75`
# - If on A100 GPU, configure CMake with: `-DCMAKE_CUDA_ARCHITECTURES=80`
# - If on L4 GPU, configure CMake with: `-DCMAKE_CUDA_ARCHITECTURES=89`

# Refer to compute capabilities here: https://developer.nvidia.com/cuda-gpus
