#include "brightness_avx.h"
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>

BrightnessAVX::BrightnessAVX(float alpha, int beta)
    : alpha(alpha), beta(beta) {}

std::string BrightnessAVX::get_name() const {
    return "Brightness/Contrast Filter (OpenMP + AVX2)";
}

void BrightnessAVX::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.empty()) {
        return;
    }

    outputImage.create(inputImage.rows, inputImage.cols, inputImage.type());

    __m256 v_alpha = _mm256_set1_ps(alpha);
    __m256 v_beta = _mm256_set1_ps(static_cast<float>(beta));

    const int rows = inputImage.rows;
    const int total_cols = inputImage.cols * inputImage.channels();

#pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        const uchar* ptr_in = inputImage.ptr<uchar>(row);
        uchar* ptr_out = outputImage.ptr<uchar>(row);

        int col = 0;

        for (; col <= total_cols - 8; col += 8) {
            __m128i v_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_in + col));

            __m256i v_int32 = _mm256_cvtepu8_epi32(v_u8);

            __m256 v_float = _mm256_cvtepi32_ps(v_int32);

            __m256 v_res = _mm256_mul_ps(v_float, v_alpha);

            v_res = _mm256_add_ps(v_res, v_beta);

            __m256i v_res_i = _mm256_cvtps_epi32(v_res);

            __m128i v_lo = _mm256_castsi256_si128(v_res_i); // Lower 4 ints
            __m128i v_hi = _mm256_extracti128_si256(v_res_i, 1); // Upper 4 ints

            __m128i v_u16 = _mm_packus_epi32(v_lo, v_hi);

            __m128i v_u8_res = _mm_packus_epi16(v_u16, v_u16);

            _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr_out + col), v_u8_res);
        }

        for (; col < total_cols; ++col) {
            float val = ptr_in[col];
            float res = val * alpha + beta;
            ptr_out[col] = cv::saturate_cast<uchar>(res);
        }
    }
}

