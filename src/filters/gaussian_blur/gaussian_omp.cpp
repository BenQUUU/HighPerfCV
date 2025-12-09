#include "gaussian_omp.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <stdexcept>

GaussianOpenMP::GaussianOpenMP(int kernel_size, float sigma)
    : kernel_size(kernel_size), sigma(sigma)
{
    if (kernel_size % 2 == 0) throw std::invalid_argument("Kernel size must be odd");
    generateKernel();
}

std::string GaussianOpenMP::get_name() const {
    return "Gaussian Blur (Separable + OpenMP)";
}

void GaussianOpenMP::generateKernel() {
    const int r = kernel_size / 2;
    kernel.resize(kernel_size);
    float sum = 0.0f;
    const float sigma2 = 2.0f * sigma * sigma;

    for (int x = -r; x <= r; ++x) {
        const float val = std::exp(-(x * x) / sigma2);
        kernel[x + r] = val;
        sum += val;
    }

    for (float& k : kernel) k /= sum;
}

void GaussianOpenMP::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;

    cv::Mat temp(input.rows, input.cols, CV_32FC3);

    output.create(input.rows, input.cols, input.type());

    const int r = kernel_size / 2;
    const int rows = input.rows;
    const int cols = input.cols;

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const uchar* ptr_in = input.ptr<uchar>(y);
        cv::Vec3f* ptr_temp = temp.ptr<cv::Vec3f>(y);

        for (int x = 0; x < cols; ++x) {
            float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;

            for (int k = -r; k <= r; ++k) {
                int nx = std::min(std::max(x + k, 0), cols - 1);

                float weight = kernel[k + r];

                int idx = nx * 3;
                sum_b += ptr_in[idx + 0] * weight;
                sum_g += ptr_in[idx + 1] * weight;
                sum_r += ptr_in[idx + 2] * weight;
            }

            ptr_temp[x] = cv::Vec3f(sum_b, sum_g, sum_r);
        }
    }

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        uchar* ptr_out = output.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
            float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;

            for (int k = -r; k <= r; ++k) {
                int ny = std::min(std::max(y + k, 0), rows - 1);

                const cv::Vec3f* ptr_temp_row = temp.ptr<cv::Vec3f>(ny);
                cv::Vec3f pixel = ptr_temp_row[x];

                float weight = kernel[k + r];

                sum_b += pixel[0] * weight;
                sum_g += pixel[1] * weight;
                sum_r += pixel[2] * weight;
            }

            ptr_out[x * 3 + 0] = cv::saturate_cast<uchar>(sum_b);
            ptr_out[x * 3 + 1] = cv::saturate_cast<uchar>(sum_g);
            ptr_out[x * 3 + 2] = cv::saturate_cast<uchar>(sum_r);
        }
    }
}
