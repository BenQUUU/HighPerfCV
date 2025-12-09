#include "gaussian_avx.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

GaussianAVX::GaussianAVX(int kernel_size, float sigma)
    : kernel_size(kernel_size), sigma(sigma) {
    if (kernel_size % 2 == 0)
        throw std::invalid_argument("Kernel size must be odd");
    generateKernel();
}

std::string GaussianAVX::get_name() const {
    return "Gaussian Blur (Separable AVX2 + OpenMP)";
}

void GaussianAVX::generateKernel() {
    int r = kernel_size / 2;
    kernel.resize(kernel_size);
    float sum = 0.0f;
    float sigma2 = 2.0f * sigma * sigma;

    for (int x = -r; x <= r; ++x) {
        float val = std::exp(-(x * x) / sigma2);
        kernel[x + r] = val;
        sum += val;
    }
    for (float& k : kernel)
        k /= sum;
}

void GaussianAVX::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty())
        return;

    cv::Mat temp(input.rows, input.cols, CV_32FC3);
    output.create(input.rows, input.cols, input.type());

    int r = kernel_size / 2;
    int rows = input.rows;
    int cols = input.cols;

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

    int total_floats = cols * 3;

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        uchar* ptr_out = output.ptr<uchar>(y);

        int x = 0;
        for (; x <= total_floats - 8; x += 8) {
            __m256 v_sum = _mm256_setzero_ps();

            for (int k = -r; k <= r; ++k) {
                int ny = std::min(std::max(y + k, 0), rows - 1);

                __m256 v_weight = _mm256_set1_ps(kernel[k + r]);

                const float* ptr_temp_row = temp.ptr<float>(ny);
                __m256 v_val = _mm256_loadu_ps(ptr_temp_row + x);

                v_sum = _mm256_fmadd_ps(v_val, v_weight, v_sum);
            }

            __m256i v_res_i = _mm256_cvtps_epi32(v_sum);

            __m128i v_lo = _mm256_castsi256_si128(v_res_i);
            __m128i v_hi = _mm256_extracti128_si256(v_res_i, 1);
            __m128i v_u16 = _mm_packus_epi32(v_lo, v_hi);

            __m128i v_u8 = _mm_packus_epi16(v_u16, v_u16);

            _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr_out + x), v_u8);
        }

        for (; x < total_floats; ++x) {
            float sum = 0.0f;
            int pixel_idx = x / 3;
            int channel = x % 3;

            for (int k = -r; k <= r; ++k) {
                int ny = std::min(std::max(y + k, 0), rows - 1);
                const float* ptr_temp_row = temp.ptr<float>(ny);
                sum += ptr_temp_row[x] * kernel[k + r];
            }
            ptr_out[x] = cv::saturate_cast<uchar>(sum);
        }
    }
}
