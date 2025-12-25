#include "median_cuda.h"
#include "median_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

MedianCUDA::MedianCUDA(int kernel_size) : kernel_size(kernel_size) {
    if (kernel_size != 3) {
        throw std::invalid_argument("MedianCUDA supports only 3x3 kernel currently.");
    }
}

std::string MedianCUDA::get_name() const {
    return "Median Filter (CUDA Optimized 3x3)";
}

void MedianCUDA::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;
    if (input.channels() != 3) throw std::invalid_argument("MedianCUDA supports only BGR images.");

    output.create(input.rows, input.cols, input.type());

    size_t size_bytes = input.rows * input.cols * 3;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    try {
        cudaMalloc(&d_input, size_bytes);
        cudaMalloc(&d_output, size_bytes);

        cudaMemcpy(d_input, input.data, size_bytes, cudaMemcpyHostToDevice);

        launchMedian3x3(d_input, d_output, input.cols, input.rows);

        cudaMemcpy(output.data, d_output, size_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw;
    }
}