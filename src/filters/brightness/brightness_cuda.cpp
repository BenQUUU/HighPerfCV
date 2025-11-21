#include "brightness_cuda.h"
#include "brightness_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

inline void checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " (CUDA Error: " + cudaGetErrorString(err) + ")");
    }
}

BrightnessCUDA::BrightnessCUDA(float alpha, int beta)
    : alpha(alpha), beta(beta) {}

std::string BrightnessCUDA::get_name() const {
    return "Brightness (CUDA - Manual)";
}

void BrightnessCUDA::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.channels() != 3) {
        throw std::invalid_argument("Incorrect number of channels");
    }

    outputImage.create(inputImage.rows, inputImage.cols, inputImage.type());

    size_t size_bytes = inputImage.step * inputImage.rows;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    try {
        checkCudaErr(cudaMalloc(&d_input, size_bytes), "Alloc input");
        checkCudaErr(cudaMalloc(&d_output, size_bytes), "Alloc output");

        checkCudaErr(cudaMemcpy(d_input, inputImage.data, size_bytes, cudaMemcpyHostToDevice), "Memcpy H2D");

        launchBrightnessKernel(d_input, d_output, inputImage.cols, inputImage.rows, inputImage.step, outputImage.step, alpha, beta);

        checkCudaErr(cudaMemcpy(outputImage.data, d_output, size_bytes, cudaMemcpyDeviceToHost), "Memcpy D2D");

        cudaFree(d_input);
        cudaFree(d_output);
    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw;
    }
}
