#include "sobel_avx.h"
#include <immintrin.h>
#include <vector>
#include <omp.h>

std::string SobelAVX::get_name() const {
    return "Sobel Edge Detection (AVX2 Vectorized)";
}

void SobelAVX::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty())
        return;
    output.create(input.rows, input.cols, input.type());

    int rows = input.rows;
    int cols = input.cols;
    int channels = input.channels();

    int row_len = cols * channels;

    int cn = channels;

    memset(output.ptr<uchar>(0), 0, row_len);
    memset(output.ptr<uchar>(rows - 1), 0, row_len);

#pragma omp parallel for schedule(static)
    for (int y = 1; y < rows - 1; ++y) {
        const uchar* prev = input.ptr<uchar>(y - 1);
        const uchar* curr = input.ptr<uchar>(y);
        const uchar* next = input.ptr<uchar>(y + 1);
        uchar* out = output.ptr<uchar>(y);

        for (int k = 0; k < cn; ++k)
            out[k] = 0;

        int x = cn;

        for (; x <= row_len - cn - 8; x += 8) {
            __m128i raw_tl = _mm_loadu_si64(prev + x - cn);  // Top-Left
            __m128i raw_ml = _mm_loadu_si64(curr + x - cn);  // Mid-Left
            __m128i raw_bl = _mm_loadu_si64(next + x - cn);  // Bot-Left

            __m256i tl = _mm256_cvtepu8_epi16(raw_tl);
            __m256i ml = _mm256_cvtepu8_epi16(raw_ml);
            __m256i bl = _mm256_cvtepu8_epi16(raw_bl);

            __m128i raw_tr = _mm_loadu_si64(prev + x + cn);  // Top-Right
            __m128i raw_mr = _mm_loadu_si64(curr + x + cn);  // Mid-Right
            __m128i raw_br = _mm_loadu_si64(next + x + cn);  // Bot-Right

            __m256i tr = _mm256_cvtepu8_epi16(raw_tr);
            __m256i mr = _mm256_cvtepu8_epi16(raw_mr);
            __m256i br = _mm256_cvtepu8_epi16(raw_br);

            __m128i raw_tc = _mm_loadu_si64(prev + x);  // Top-Center
            __m128i raw_bc = _mm_loadu_si64(next + x);  // Bot-Center

            __m256i tc = _mm256_cvtepu8_epi16(raw_tc);
            __m256i bc = _mm256_cvtepu8_epi16(raw_bc);

            // Gx = (tr + 2*mr + br) - (tl + 2*ml + bl)
            // Gy = (bl + 2*bc + br) - (tl + 2*tc + tr)

            __m256i sum_r = _mm256_add_epi16(tr, _mm256_add_epi16(_mm256_slli_epi16(mr, 1), br));
            __m256i sum_l = _mm256_add_epi16(tl, _mm256_add_epi16(_mm256_slli_epi16(ml, 1), bl));

            __m256i gx_i16 = _mm256_sub_epi16(sum_r, sum_l);

            __m256i sum_b = _mm256_add_epi16(bl, _mm256_add_epi16(_mm256_slli_epi16(bc, 1), br));
            __m256i sum_t = _mm256_add_epi16(tl, _mm256_add_epi16(_mm256_slli_epi16(tc, 1), tr));

            __m256i gy_i16 = _mm256_sub_epi16(sum_b, sum_t);

            __m256i gx_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(gx_i16));
            __m256i gy_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(gy_i16));

            __m256 gx_ps = _mm256_cvtepi32_ps(gx_i32);
            __m256 gy_ps = _mm256_cvtepi32_ps(gy_i32);

            __m256 g2 = _mm256_add_ps(_mm256_mul_ps(gx_ps, gx_ps), _mm256_mul_ps(gy_ps, gy_ps));

            __m256 mag_ps = _mm256_sqrt_ps(g2);

            __m256i mag_i32 = _mm256_cvtps_epi32(mag_ps);

            __m256i mag_i16 = _mm256_packs_epi32(mag_i32, _mm256_setzero_si256());

            __m256i mag_u8_vec = _mm256_packus_epi16(mag_i16, _mm256_setzero_si256());

            _mm_storel_epi64((__m128i*)(out + x), _mm256_castsi256_si128(mag_u8_vec));
        }

        for (; x < row_len - cn; ++x) {
            int c_idx = x % channels;

            int idx = x;
            int gx = -1 * prev[idx - cn] + 1 * prev[idx + cn] - 2 * curr[idx - cn] + 2 * curr[idx + cn] - 1 * next[idx - cn] + 1 * next[idx + cn];

            int gy = -1 * prev[idx - cn] - 2 * prev[idx] - 1 * prev[idx + cn] + 1 * next[idx - cn] + 2 * next[idx] + 1 * next[idx + cn];

            float val = std::sqrt(static_cast<float>(gx * gx + gy * gy));
            out[x] = cv::saturate_cast<uchar>(val);
        }

        for (int k = 0; k < cn; ++k)
            out[row_len - cn + k] = 0;
    }
}