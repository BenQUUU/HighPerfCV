#include "median_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ void sort2(unsigned char& a, unsigned char& b) {
    unsigned char min_val = min(a, b);
    unsigned char max_val = max(a, b);
    a = min_val;
    b = max_val;
}

__global__ void median3x3Kernel(const uchar3* input, uchar3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int idx = y * width + x;
        int prev_row = idx - width;
        int next_row = idx + width;

        uchar3 p0 = input[prev_row - 1];
        uchar3 p1 = input[prev_row];
        uchar3 p2 = input[prev_row + 1];

        uchar3 p3 = input[idx - 1];
        uchar3 p4 = input[idx];
        uchar3 p5 = input[idx + 1];

        uchar3 p6 = input[next_row - 1];
        uchar3 p7 = input[next_row];
        uchar3 p8 = input[next_row + 1];

        sort2(p0.x, p1.x); sort2(p1.x, p2.x); sort2(p0.x, p1.x);
        sort2(p3.x, p4.x); sort2(p4.x, p5.x); sort2(p3.x, p4.x);
        sort2(p6.x, p7.x); sort2(p7.x, p8.x); sort2(p6.x, p7.x);
        
        unsigned char min_of_maxs_x = min(p2.x, min(p5.x, p8.x));
        unsigned char max_of_mins_x = max(p0.x, max(p3.x, p6.x));
        
        sort2(p1.x, p4.x); sort2(p4.x, p7.x); sort2(p1.x, p4.x);
        unsigned char med_of_meds_x = p4.x;
        
        unsigned char result_x = max(max_of_mins_x, min(med_of_meds_x, min_of_maxs_x));

        sort2(p0.y, p1.y); sort2(p1.y, p2.y); sort2(p0.y, p1.y);
        sort2(p3.y, p4.y); sort2(p4.y, p5.y); sort2(p3.y, p4.y);
        sort2(p6.y, p7.y); sort2(p7.y, p8.y); sort2(p6.y, p7.y);

        unsigned char min_of_maxs_y = min(p2.y, min(p5.y, p8.y));
        unsigned char max_of_mins_y = max(p0.y, max(p3.y, p6.y));

        sort2(p1.y, p4.y); sort2(p4.y, p7.y); sort2(p1.y, p4.y);
        unsigned char med_of_meds_y = p4.y;
        
        sort2(min_of_maxs_y, max_of_mins_y);
        sort2(max_of_mins_y, med_of_meds_y);
        
        unsigned char result_y = max_of_mins_y;

        sort2(p0.z, p1.z); sort2(p1.z, p2.z); sort2(p0.z, p1.z);
        sort2(p3.z, p4.z); sort2(p4.z, p5.z); sort2(p3.z, p4.z);
        sort2(p6.z, p7.z); sort2(p7.z, p8.z); sort2(p6.z, p7.z);

        unsigned char min_of_maxs_z = min(p2.z, min(p5.z, p8.z));
        unsigned char max_of_mins_z = max(p0.z, max(p3.z, p6.z));

        sort2(p1.z, p4.z); sort2(p4.z, p7.z); sort2(p1.z, p4.z);
        unsigned char med_of_meds_z = p4.z;

        sort2(min_of_maxs_z, max_of_mins_z);
        sort2(max_of_mins_z, med_of_meds_z);
        unsigned char result_z = max_of_mins_z;

        output[idx] = make_uchar3(result_x, result_y, result_z);

    } else {
        if (x < width && y < height) {
            output[y * width + x] = input[y * width + x];
        }
    }
}

void launchMedian3x3(const unsigned char* d_input, 
                     unsigned char* d_output, 
                     int width, int height) 
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, 
                (height + threads.y - 1) / threads.y);

    median3x3Kernel<<<blocks, threads>>>(
        (const uchar3*)d_input,
        (uchar3*)d_output,
        width, height
    );
    cudaDeviceSynchronize();
}