#include "gaussian_base.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

GaussianBase::GaussianBase(int kernel_size, float sigma)
    : kernel_size(kernel_size), sigma(sigma)
{
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("The kernel size must be odd (e.g. 3, 5, 7)!");
    }
    generateKernel();
}

std::string GaussianBase::get_name() const {
    return "Gaussian Blur (Base 2D C++)";
}

void GaussianBase::generateKernel() {
    const int r = kernel_size / 2;
    kernel.resize(kernel_size * kernel_size);

    float sum = 0.0f;
    const float sigma2 = 2.0f * sigma * sigma;

    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r; ++x) {
            const float val = std::exp(-(x*x + y*y) / sigma2);

            kernel[(y + r) * kernel_size + (x + r)] = val;
            sum += val;
        }
    }

    for (float& k : kernel) {
        k /= sum;
    }
}

void GaussianBase::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;

    if (input.channels() != 3) {
        throw std::invalid_argument("GaussianBase only supports BGR images");
    }

    output.create(input.rows, input.cols, input.type());

    const int r = kernel_size / 2;
    const int rows = input.rows;
    const int cols = input.cols;

    for (int y = 0; y < rows; ++y) {
        uchar* ptr_out = output.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
            float sum_b = 0.0f;
            float sum_g = 0.0f;
            float sum_r = 0.0f;

            for (int ky = -r; ky <= r; ++ky) {
                const int ny = std::min(std::max(y + ky, 0), rows - 1);

                const uchar* ptr_in_row = input.ptr<uchar>(ny);

                for (int kx = -r; kx <= r; ++kx) {
                    const int nx = std::min(std::max(x + kx, 0), cols - 1);

                    const float weight = kernel[(ky + r) * kernel_size + (kx + r)];

                    const int pixel_idx = nx * 3;

                    sum_b += ptr_in_row[pixel_idx + 0] * weight;
                    sum_g += ptr_in_row[pixel_idx + 1] * weight;
                    sum_r += ptr_in_row[pixel_idx + 2] * weight;
                }
            }

            ptr_out[x * 3 + 0] = cv::saturate_cast<uchar>(sum_b);
            ptr_out[x * 3 + 1] = cv::saturate_cast<uchar>(sum_g);
            ptr_out[x * 3 + 2] = cv::saturate_cast<uchar>(sum_r);
        }
    }
}
