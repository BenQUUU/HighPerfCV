#include "grayscale_kernel.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void grayscaleKernel_Raw(const uchar3* input,
                                    unsigned char* output,
                                    int width, int height,
                                    size_t input_pitch, size_t output_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const char* row_in_char = (const char*)input + y * input_pitch;
        char* row_out_char = (char*)output + y * output_pitch;

        const uchar3* row_in_pixels = (const uchar3*)row_in_char;
        uchar* row_out_pixels = (uchar*)row_out_char;

        uchar3 bgr = row_in_pixels[x];

        float gray_val = 0.114f * bgr.x + 0.587f * bgr.y + 0.299f * bgr.z;

        row_out_pixels[x] = static_cast<uchar>(gray_val);
    }
}

void launchGrayscaleKernel(const unsigned char* d_input,
                             unsigned char* d_output,
                             int width, int height,
                             size_t input_pitch, size_t output_pitch)
{
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    grayscaleKernel_Raw<<<numBlocks, threadsPerBlock>>>(
        (const uchar3*)d_input,
        d_output,
        width, height,
        input_pitch, output_pitch
    );

    cudaDeviceSynchronize();
}