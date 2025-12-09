#include <cuda_runtime.h>
#include <stdio.h>
#include "gaussian_kernel.h"

__constant__ float c_kernel[MAX_KERNEL_SIZE];

void uploadGaussianKernelToConstant(const float* host_kernel, int size) {
    if (size > MAX_KERNEL_SIZE) {
        printf("Error: Kernel size %d exceeds limit %d\n", size, MAX_KERNEL_SIZE);
        return;
    }

    cudaMemcpyToSymbol(c_kernel, host_kernel, size * sizeof(float));
}

__global__ void rowKernel(const uchar3* input, float3* output, int width, int height, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);

        for (int k = -r; k <= r; ++k) {
            int nx = min(max(x + k, 0), width - 1);

            uchar3 pixel = input[y * width + nx];

            float weight = c_kernel[k + r];

            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
        }

        output[y * width + x] = sum;
    }
}

__global__ void colKernel(const float3* input, uchar3* output, int width, int height, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);

        for (int k = -r; k <= r; ++k) {
            int ny = min(max(y + k, 0), height - 1);

            float3 pixel = input[ny * width + x];
            float weight = c_kernel[k + r];

            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
        }

        output[y * width + x] = make_uchar3(
            (unsigned char)fminf(fmaxf(sum.x, 0.0f), 255.0f),
            (unsigned char)fminf(fmaxf(sum.y, 0.0f), 255.0f),
            (unsigned char)fminf(fmaxf(sum.z, 0.0f), 255.0f));
    }
}

void launchGaussianSeparable(const unsigned char* d_input,
                             unsigned char* d_output,
                             float* d_temp,
                             int width,
                             int height,
                             int kernel_radius) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    rowKernel<<<blocks, threads>>>(
        (const uchar3*)d_input,
        (float3*)d_temp,
        width, height,
        kernel_radius);

    cudaDeviceSynchronize();

    colKernel<<<blocks, threads>>>(
        (const float3*)d_temp,
        (uchar3*)d_output,
        width, height,
        kernel_radius);

    cudaDeviceSynchronize();
}