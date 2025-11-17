#include "grayscale_avx.h"
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>

std::string GrayscaleAVX::get_name() const {
    return "Grayscale (AVX2 + OpenMP)";
}

void GrayscaleAVX::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.channels() != 3) {
        throw std::invalid_argument("The input image for GrayscaleAVX must be 3-channel (BGR)");
    }

    outputImage.create(inputImage.rows, inputImage.cols, CV_8UC1);

    const __m256 b_mul = _mm256_set1_ps(0.114f);
    const __m256 g_mul = _mm256_set1_ps(0.587f);
    const __m256 r_mul = _mm256_set1_ps(0.299f);

    #pragma omp parallel for
    for (int row = 0; row < inputImage.rows; ++row) {
        const cv::Vec3b* ptr_in = inputImage.ptr<cv::Vec3b>(row);
        uchar* ptr_out = outputImage.ptr<uchar>(row);

        int stop_c = (inputImage.cols / 8) * 8;

        for (int col = 0; col < stop_c; col += 8) {
            float b_vals[8], g_vals[8], r_vals[8];
            for (int i = 0; i < 8; ++i) {
                b_vals[i] = static_cast<float>(ptr_in[col + i][0]);
                g_vals[i] = static_cast<float>(ptr_in[col + i][1]);
                r_vals[i] = static_cast<float>(ptr_in[col + i][2]);
            }

            __m256 b_ps = _mm256_loadu_ps(b_vals);
            __m256 g_ps = _mm256_loadu_ps(g_vals);
            __m256 r_ps = _mm256_loadu_ps(r_vals);

            __m256 b_part = _mm256_mul_ps(b_ps, b_mul);
            __m256 g_part = _mm256_mul_ps(g_ps, g_mul);
            __m256 r_part = _mm256_mul_ps(r_ps, r_mul);

            __m256 gray_ps = _mm256_add_ps(b_part, _mm256_add_ps(g_part, r_part));

            __m256i gray_epi32 = _mm256_cvtps_epi32(gray_ps);

            int gray_ints[8];
            _mm256_storeu_si256((__m256i*)gray_ints, gray_epi32);
            for(int i = 0; i < 8; ++i) {
                ptr_out[col + i] = static_cast<uchar>(gray_ints[i]);
            }
        }

        for (int c = stop_c; c < inputImage.cols; ++c) {
            const cv::Vec3b& pixel_in = ptr_in[c];
            const float gray_val = 0.114f * pixel_in[0] + 0.587f * pixel_in[1] + 0.299f * pixel_in[2];
            ptr_out[c] = static_cast<uchar>(gray_val);
        }
    }
}
