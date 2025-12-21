#include "median_avx.h"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

#define SORT2(a, b)                          \
    {                                        \
        __m256i min = _mm256_min_epu8(a, b); \
        __m256i max = _mm256_max_epu8(a, b); \
        a = min;                             \
        b = max;                             \
    }

MedianAVX::MedianAVX(int kernel_size) : kernel_size(kernel_size) {
    if (kernel_size % 2 == 0 || kernel_size < 1) {
        throw std::invalid_argument("Median kernel size must be odd.");
    }
}

std::string MedianAVX::get_name() const {
    if (kernel_size == 3) return "Median Filter (AVX2 Optimized 3x3)";
    return "Median Filter (AVX2 Fallback to Scalar)";
}

void processScalar(const cv::Mat& input, cv::Mat& output, int kernel_size, int y, int x_start, int x_end) {
    const int r = kernel_size / 2;
    const int channels = input.channels();
    const int rows = input.rows;
    const int cols = input.cols;
    std::vector<uchar> window(kernel_size * kernel_size);

    for (int x = x_start; x < x_end; ++x) {
        for (int c = 0; c < channels; ++c) {
            int k = 0;
            for (int ky = -r; ky <= r; ++ky) {
                int ny = std::min(std::max(y + ky, 0), rows - 1);
                const uchar* ptr_in = input.ptr<uchar>(ny);
                for (int kx = -r; kx <= r; ++kx) {
                    int nx = std::min(std::max(x + kx, 0), cols - 1);
                    window[k++] = ptr_in[nx * channels + c];
                }
            }
            std::sort(window.begin(), window.end());
            output.ptr<uchar>(y)[x * channels + c] = window[window.size() / 2];
        }
    }
}

void MedianAVX::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;
    output.create(input.rows, input.cols, input.type());

    const int rows = input.rows;
    const int cols = input.cols;
    const int channels = input.channels();

    const int row_len = cols * channels;

    if (kernel_size != 3) {
        for(int y=0; y<rows; ++y) processScalar(input, output, kernel_size, y, 0, cols);
        return;
    }

    for (int y = 1; y < rows - 1; ++y) {
        const uchar* prev = input.ptr<uchar>(y - 1);
        const uchar* curr = input.ptr<uchar>(y);
        const uchar* next = input.ptr<uchar>(y + 1);
        uchar* out = output.ptr<uchar>(y);

        int i = channels;

        for (; i <= row_len - 32 - channels; i += 32) {
            __m256i p0 = _mm256_loadu_si256((const __m256i*)(prev + i - channels));
            __m256i p1 = _mm256_loadu_si256((const __m256i*)(prev + i));
            __m256i p2 = _mm256_loadu_si256((const __m256i*)(prev + i + channels));

            __m256i p3 = _mm256_loadu_si256((const __m256i*)(curr + i - channels));
            __m256i p4 = _mm256_loadu_si256((const __m256i*)(curr + i));
            __m256i p5 = _mm256_loadu_si256((const __m256i*)(curr + i + channels));

            __m256i p6 = _mm256_loadu_si256((const __m256i*)(next + i - channels));
            __m256i p7 = _mm256_loadu_si256((const __m256i*)(next + i));
            __m256i p8 = _mm256_loadu_si256((const __m256i*)(next + i + channels));

            SORT2(p0, p1); SORT2(p1, p2); SORT2(p0, p1);

            SORT2(p3, p4); SORT2(p4, p5); SORT2(p3, p4);

            SORT2(p6, p7); SORT2(p7, p8); SORT2(p6, p7);

            __m256i min_of_maxs = _mm256_min_epu8(p2, _mm256_min_epu8(p5, p8));

            __m256i max_of_mins = _mm256_max_epu8(p0, _mm256_max_epu8(p3, p6));

            SORT2(p1, p4); SORT2(p4, p7); SORT2(p1, p4);
            __m256i med_of_mids = p4;

            SORT2(min_of_maxs, max_of_mins);
            SORT2(max_of_mins, med_of_mids);
            SORT2(min_of_maxs, max_of_mins);

            __m256i result = max_of_mins;

            _mm256_storeu_si256((__m256i*)(out + i), result);
        }

        processScalar(input, output, kernel_size, y, i / channels, cols);

        processScalar(input, output, kernel_size, y, 0, 1);
    }

    processScalar(input, output, kernel_size, 0, 0, cols);
    processScalar(input, output, kernel_size, rows - 1, 0, cols);
}
