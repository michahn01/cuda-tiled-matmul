# Tiled Matrix Multiplication in CUDA

Just a small, practice implementation of matrix multiplication in CUDA.

## Tiling

GPU compute performance can often be bottlenecked by global memory bandwidth, preventing programs from reaching peak computational throughput (FLOPS) available on the GPU. Tiling is a technique that mitigates this by leveraging the much faster, lower-latency shared memory available on each streaming multiprocessor (SM) on the GPU. Because shared memory is block-scoped, threads within a block collaboratively store/load from this shared memory, treating it like a programmer-managed shared local cache to significantly reduce VRAM accesses through data reuse. For problems like matrix multiplication, where there is large overlap between the data accessed by threads from global memory, tiling can provide dramatic performance improvements.

This repo is a small example that uses tiling to speed up matrix multiplication in CUDA.

## Platforms

I was able to verify successful compilation and execution on:
- NVIDIA T4 GPU
- NVIDIA A100 GPU
- NVIDIA L4 GPU

Note: depeding on your setup, compiling and running on a platform may require specifying the architecture via compile flags. When configuring CMake (e.g., with `cmake -S . -B build` or similar), you may have to append a `-DCMAKE_CUDA_ARCHITECTURES=XX` flag at the end. For example:

- If on T4, add this: `-DCMAKE_CUDA_ARCHITECTURES=75`
- If on A100, add this: `-DCMAKE_CUDA_ARCHITECTURES=80`
- If on L4, add this: `-DCMAKE_CUDA_ARCHITECTURES=89`

Refer to compute capabilities here: https://developer.nvidia.com/cuda-gpus
