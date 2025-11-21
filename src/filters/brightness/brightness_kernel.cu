#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "brightness_kernel.h"

__global__ void brightnessKernel(const uchar3* input,
                                 uchar3* output,
                                 int width,
                                 int height,
                                 size_t input_pitch,
                                 size_t output_pitch,
                                 float alpha,
                                 int beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const uchar3* row_in = (const uchar3*)((const char*)input + y * input_pitch);
        uchar3* row_out = (uchar3*)((char*)output + y * output_pitch);

        uchar3 pixel = row_in[x];

        float b = alpha * pixel.x + beta;
        float g = alpha * pixel.y + beta;
        float r = alpha * pixel.z + beta;

        b = fminf(fmaxf(b, 0.0f), 255.0f);
        g = fminf(fmaxf(g, 0.0f), 255.0f);
        r = fminf(fmaxf(r, 0.0f), 255.0f);

        row_out[x] = make_uchar3((uchar)b, (uchar)g, (uchar)r);
    }
}

void launchBrightnessKernel(const unsigned char* d_input,
                            unsigned char* d_output,
                            int width,
                            int height,
                            size_t input_pitch,
                            size_t output_pitch,
                            float alpha,
                            int beta) {
    const dim3 threads(16, 16);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    brightnessKernel<<<blocks, threads>>>(
        (const uchar3*)d_input,
        (uchar3*)d_output,
        width, height,
        input_pitch, output_pitch,
        alpha, beta);

    cudaDeviceSynchronize();
}