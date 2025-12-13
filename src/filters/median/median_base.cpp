#include "median_base.h"
#include <algorithm>
#include <vector>
#include <stdexcept>

MedianBase::MedianBase(int kernel_size) : kernel_size(kernel_size) {
    if (kernel_size % 2 == 0 || kernel_size < 1) {
        throw std::invalid_argument("Kernel size must be a positive odd integer.");
    }
}

std::string MedianBase::get_name() const {
    return "Median Filter (Base C++)";
}

void MedianBase::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.empty()) return;

    outputImage.create(inputImage.rows, inputImage.cols, inputImage.type());

    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int channels = inputImage.channels();
    const int r = kernel_size / 2;

    std::vector<uchar> window;
    window.reserve(kernel_size * kernel_size);

    for (int y = 0; y < rows; ++y) {
        uchar* ptr_out = outputImage.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                window.clear();

                for (int ky = -r; ky <= r; ++ky) {
                    for (int kx = -r; kx <= r; ++kx) {
                        int ny = std::min(std::max(y + ky, 0), rows - 1);
                        int nx = std::min(std::max(x + kx, 0), cols - 1);

                        const uchar* ptr_in_row = inputImage.ptr<uchar>(ny);
                        uchar val = ptr_in_row[nx * channels + c];

                        window.push_back(val);
                    }
                }
                std::sort(window.begin(), window.end());

                uchar median = window[window.size() / 2];

                ptr_out[x * channels + c] = median;
            }
        }
    }
}
