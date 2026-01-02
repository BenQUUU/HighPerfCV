#include "median_neon.h"
#include <arm_neon.h>
#include <vector>
#include <algorithm>
#include <stdexcept>

#define SORT2(a, b) { \
    uint8x16_t min = vminq_u8(a, b); \
    uint8x16_t max = vmaxq_u8(a, b); \
    a = min; \
    b = max; \
}

MedianNEON::MedianNEON(int kernel_size) : kernel_size(kernel_size) {
    if (kernel_size % 2 == 0 || kernel_size < 1) {
        throw std::invalid_argument("Median kernel size must be odd.");
    }
}

std::string MedianNEON::get_name() const {
    if (kernel_size == 3) return "Median Filter (NEON Optimized 3x3)";
    return "Median Filter (NEON Fallback to Scalar)";
}

void processScalarNeon(const cv::Mat& input, cv::Mat& output, int kernel_size, int y, int x_start, int x_end) {
    int r = kernel_size / 2;
    int channels = input.channels();
    int rows = input.rows;
    int cols = input.cols;
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

void MedianNEON::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;
    output.create(input.rows, input.cols, input.type());

    int rows = input.rows;
    int cols = input.cols;
    int channels = input.channels();
    
    int row_len = cols * channels;

    if (kernel_size != 3) {
        for(int y=0; y<rows; ++y) processScalarNeon(input, output, kernel_size, y, 0, cols);
        return;
    }
    
    for (int y = 1; y < rows - 1; ++y) {
        const uchar* prev = input.ptr<uchar>(y - 1);
        const uchar* curr = input.ptr<uchar>(y);
        const uchar* next = input.ptr<uchar>(y + 1);
        uchar* out = output.ptr<uchar>(y);

        int i = channels;

        for (; i <= row_len - 16 - channels; i += 16) {
            uint8x16_t p0 = vld1q_u8(prev + i - channels);
            uint8x16_t p3 = vld1q_u8(curr + i - channels);
            uint8x16_t p6 = vld1q_u8(next + i - channels);

            uint8x16_t p1 = vld1q_u8(prev + i);
            uint8x16_t p4 = vld1q_u8(curr + i);
            uint8x16_t p7 = vld1q_u8(next + i);

            uint8x16_t p2 = vld1q_u8(prev + i + channels);
            uint8x16_t p5 = vld1q_u8(curr + i + channels);
            uint8x16_t p8 = vld1q_u8(next + i + channels);

            SORT2(p0, p1); SORT2(p1, p2); SORT2(p0, p1);
            
            SORT2(p3, p4); SORT2(p4, p5); SORT2(p3, p4);
            
            SORT2(p6, p7); SORT2(p7, p8); SORT2(p6, p7);

            uint8x16_t min_of_maxs = vminq_u8(p2, vminq_u8(p5, p8));

            uint8x16_t max_of_mins = vmaxq_u8(p0, vmaxq_u8(p3, p6));

            SORT2(p1, p4); SORT2(p4, p7); SORT2(p1, p4);
            uint8x16_t med_of_mids = p4;

            SORT2(min_of_maxs, max_of_mins); 
            SORT2(max_of_mins, med_of_mids); 
            SORT2(min_of_maxs, max_of_mins);
            
            uint8x16_t result = max_of_mins;

            vst1q_u8(out + i, result);
        }

        processScalarNeon(input, output, kernel_size, y, i / channels, cols);
        
        processScalarNeon(input, output, kernel_size, y, 0, 1);
    }

    processScalarNeon(input, output, kernel_size, 0, 0, cols);
    processScalarNeon(input, output, kernel_size, rows - 1, 0, cols);
}