#pragma once
#include <vector>

#define MAX_KERNEL_SIZE 64

void uploadGaussianKernelToConstant(const float* host_kernel, int size);

void launchGaussianSeparable(const unsigned char* d_input, 
                             unsigned char* d_output,
                             float* d_temp,
                             int width, int height, 
                             int kernel_radius);