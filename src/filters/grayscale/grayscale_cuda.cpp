#include "grayscale_cuda.h"
#include "grayscale_kernel.h"
#include <stdexcept>
#include <cuda_runtime.h>

inline void checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " (CUDA Error: " + cudaGetErrorString(err) + ")");
    }
}

std::string GrayscaleCUDA::get_name() const {
    return "Grayscale (CUDA - Manual)";
}

void GrayscaleCUDA::process(const cv::Mat& input, cv::Mat& output) {

    if (input.channels() != 3) {
        throw std::invalid_argument("The input image for GrayscaleCUDA must be 3-channel (BGR)");
    }

    output.create(input.rows, input.cols, CV_8UC1);

    if (!input.isContinuous() || !output.isContinuous()) {
        throw std::runtime_error("GrayscaleCUDA (Manual) requires contiguous cv::Mat memory blocks");
    }

    int width = input.cols;
    int height = input.rows;
    size_t input_pitch = input.step;
    size_t output_pitch = output.step;

    size_t input_size_bytes = input_pitch * height;
    size_t output_size_bytes = output_pitch * height;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    try {
        checkCudaErr(cudaMalloc(&d_input, input_size_bytes), "cudaMalloc d_input");
        checkCudaErr(cudaMalloc(&d_output, output_size_bytes), "cudaMalloc d_output");

        checkCudaErr(cudaMemcpy(d_input, input.data, input_size_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy Host-to-Device");

        launchGrayscaleKernel(d_input, d_output, width, height, input_pitch, output_pitch);

        checkCudaErr(cudaMemcpy(output.data, d_output, output_size_bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy Device-to-Host");

        cudaFree(d_input);
        cudaFree(d_output);

    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw;
    }
}
