#include "gaussian_cuda.h"
#include "gaussian_kernel.h"
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>

GaussianCUDA::GaussianCUDA(int kernel_size, float sigma)
    : kernel_size(kernel_size), sigma(sigma)
{
    if (kernel_size % 2 == 0) throw std::invalid_argument("Kernel size must be odd");
    if (kernel_size > MAX_KERNEL_SIZE) throw std::invalid_argument("Kernel size too large for CUDA constant memory");

    int r = kernel_size / 2;
    kernel.resize(kernel_size);
    float sum = 0.0f;
    float sigma2 = 2.0f * sigma * sigma;
    for (int x = -r; x <= r; ++x) {
        float val = std::exp(-(x * x) / sigma2);
        kernel[x + r] = val;
        sum += val;
    }
    for (float& k : kernel) k /= sum;
}

std::string GaussianCUDA::get_name() const {
    return "Gaussian Blur (CUDA Separable + Constant Mem)";
}

void GaussianCUDA::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;
    if (input.channels() != 3) throw std::invalid_argument("Only BGR supported");

    output.create(input.rows, input.cols, input.type());

    int width = input.cols;
    int height = input.rows;
    int r = kernel_size / 2;

    uploadGaussianKernelToConstant(kernel.data(), kernel.size());

    size_t img_size_bytes = width * height * 3 * sizeof(unsigned char);
    size_t temp_size_bytes = width * height * 3 * sizeof(float);

    unsigned char *d_input = nullptr, *d_output = nullptr;
    float *d_temp = nullptr;

    try {
        cudaMalloc(&d_input, img_size_bytes);
        cudaMalloc(&d_output, img_size_bytes);
        cudaMalloc(&d_temp, temp_size_bytes);

        // Upload
        cudaMemcpy(d_input, input.data, img_size_bytes, cudaMemcpyHostToDevice);

        // Input -> Temp -> Output
        launchGaussianSeparable(d_input, d_output, d_temp, width, height, r);

        // Download
        cudaMemcpy(output.data, d_output, img_size_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_temp);

    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_temp) cudaFree(d_temp);
        throw;
    }
}
