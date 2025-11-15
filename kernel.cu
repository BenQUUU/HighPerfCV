#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_kernel()
{
    printf("  [CUDA]: Hello from CUDA thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

void launch_test_kernel()
{
    printf("\n--- Test 2: Uruchamianie kernela CUDA ---\n");

    simple_kernel<<<1, 4>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Błąd CUDA: %s\n", cudaGetErrorString(err));
    }

    printf("--- Kernel CUDA zakończył działanie ---\n");
}