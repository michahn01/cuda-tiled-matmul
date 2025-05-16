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
