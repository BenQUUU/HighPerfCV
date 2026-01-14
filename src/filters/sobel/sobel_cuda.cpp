#include "sobel_cuda.h"
#include "sobel_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

std::string SobelCUDA::get_name() const {
    return "Sobel Edge Detection (CUDA GPU)";
}

void SobelCUDA::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;

    if (input.type() != CV_8UC3) {
        throw std::runtime_error("SobelCUDA supports only 3-channel (BGR) images.");
    }

    output.create(input.rows, input.cols, input.type());

    size_t size_bytes = input.rows * input.cols * 3;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    try {
        cudaMalloc(&d_input, size_bytes);
        cudaMalloc(&d_output, size_bytes);

        cudaMemcpy(d_input, input.data, size_bytes, cudaMemcpyHostToDevice);

        launchSobel(d_input, d_output, input.cols, input.rows);

        cudaMemcpy(output.data, d_output, size_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw;
    }
}