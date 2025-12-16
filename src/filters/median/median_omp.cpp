#include "median_omp.h"
#include <omp.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

MedianOpenMP::MedianOpenMP(int kernel_size) : kernel_size(kernel_size) {
    if (kernel_size % 2 == 0 || kernel_size < 1) {
        throw std::invalid_argument("Median kernel size must be odd.");
    }
}

std::string MedianOpenMP::get_name() const {
    return "Median Filter (OpenMP)";
}

void MedianOpenMP::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.empty())
        return;

    outputImage.create(inputImage.rows, inputImage.cols, inputImage.type());

    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int channels = inputImage.channels();
    const int r = kernel_size / 2;
    const int area = kernel_size * kernel_size;

    #pragma omp parallel
    {
        std::vector<uchar> window(area);

        #pragma omp for
        for (int y = 0; y < rows; ++y) {
            uchar* ptr_out = outputImage.ptr<uchar>(y);

            for (int x = 0; x < cols; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int k = 0;
                    for (int ky = -r; ky <= r; ++ky) {
                        int ny = std::min(std::max(y + ky, 0), rows - 1);
                        const uchar* ptr_in_row = inputImage.ptr<uchar>(ny);

                        for (int kx = -r; kx <= r; ++kx) {
                            int nx = std::min(std::max(x + kx, 0), cols - 1);
                            window[k++] = ptr_in_row[nx * channels + c];
                        }
                    }
                    std::sort(window.begin(), window.end());

                    ptr_out[x * channels + c] = window[area / 2];
                }
            }
        }
    }
}
