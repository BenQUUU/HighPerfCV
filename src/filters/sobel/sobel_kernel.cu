#include "sobel_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void sobelKernel(const uchar3* input, uchar3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {

        int idx = y * width + x;

        uchar3 t_l = input[idx - width - 1];
        uchar3 t_c = input[idx - width];
        uchar3 t_r = input[idx - width + 1];

        uchar3 m_l = input[idx - 1];

        uchar3 m_r = input[idx + 1];

        uchar3 b_l = input[idx + width - 1];
        uchar3 b_c = input[idx + width];
        uchar3 b_r = input[idx + width + 1];

        uchar3 res;

        // (Blue)
        {
            float gx = -1.0f * t_l.x + 1.0f * t_r.x
                       -2.0f * m_l.x + 2.0f * m_r.x
                       -1.0f * b_l.x + 1.0f * b_r.x;

            float gy = -1.0f * t_l.x - 2.0f * t_c.x - 1.0f * t_r.x
                       +1.0f * b_l.x + 2.0f * b_c.x + 1.0f * b_r.x;

            float mag = __fsqrt_rn(gx * gx + gy * gy);

            res.x = (unsigned char)fminf(fmaxf(mag, 0.0f), 255.0f);
        }

        // (Green)
        {
            float gx = -1.0f * t_l.y + 1.0f * t_r.y
                       -2.0f * m_l.y + 2.0f * m_r.y
                       -1.0f * b_l.y + 1.0f * b_r.y;

            float gy = -1.0f * t_l.y - 2.0f * t_c.y - 1.0f * t_r.y
                       +1.0f * b_l.y + 2.0f * b_c.y + 1.0f * b_r.y;

            float mag = __fsqrt_rn(gx * gx + gy * gy);
            res.y = (unsigned char)fminf(fmaxf(mag, 0.0f), 255.0f);
        }

        // (Red)
        {
            float gx = -1.0f * t_l.z + 1.0f * t_r.z
                       -2.0f * m_l.z + 2.0f * m_r.z
                       -1.0f * b_l.z + 1.0f * b_r.z;

            float gy = -1.0f * t_l.z - 2.0f * t_c.z - 1.0f * t_r.z
                       +1.0f * b_l.z + 2.0f * b_c.z + 1.0f * b_r.z;

            float mag = __fsqrt_rn(gx * gx + gy * gy);
            res.z = (unsigned char)fminf(fmaxf(mag, 0.0f), 255.0f);
        }

        output[idx] = res;
    }
    else if (x < width && y < height) {
        output[y * width + x] = make_uchar3(0, 0, 0);
    }
}

void launchSobel(const unsigned char* d_input,
                 unsigned char* d_output,
                 int width, int height)
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    sobelKernel<<<blocks, threads>>>(
        (const uchar3*)d_input,
        (uchar3*)d_output,
        width, height
    );
    cudaDeviceSynchronize();
}