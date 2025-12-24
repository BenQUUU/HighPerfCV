#include "grayscale_neon.h"
#include <arm_neon.h>
#include <omp.h>

std::string GrayscaleNEON::get_name() const {
    return "Grayscale Conversion (ARM NEON)";
}

void GrayscaleNEON::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;

    output.create(input.rows, input.cols, CV_8UC1);

    int rows = input.rows;
    int cols = input.cols;
    int total_pixels = rows * cols;

    if (input.isContinuous() && output.isContinuous()) {
        cols = total_pixels;
        rows = 1;
    }

    const uint8_t w_r = 77;
    const uint8_t w_g = 150;
    const uint8_t w_b = 29;

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const uint8_t* src = input.ptr<uint8_t>(y);
        uint8_t* dst = output.ptr<uint8_t>(y);

        int x = 0;
        
        for (; x <= cols - 16; x += 16) {
            uint8x16x3_t bgr = vld3q_u8(src + x * 3);

            uint8x16_t b = bgr.val[0];
            uint8x16_t g = bgr.val[1];
            uint8x16_t r = bgr.val[2];

            uint16x8_t acc_lo = vmull_u8(vget_low_u8(b), vdup_n_u8(w_b));
        
            acc_lo = vmlal_u8(acc_lo, vget_low_u8(g), vdup_n_u8(w_g));
            acc_lo = vmlal_u8(acc_lo, vget_low_u8(r), vdup_n_u8(w_r));

            uint16x8_t acc_hi = vmull_high_u8(b, vdupq_n_u8(w_b));
            acc_hi = vmlal_high_u8(acc_hi, g, vdupq_n_u8(w_g));
            acc_hi = vmlal_high_u8(acc_hi, r, vdupq_n_u8(w_r));
            
            uint8x8_t res_lo = vshrn_n_u16(acc_lo, 8);
            uint8x8_t res_hi = vshrn_n_u16(acc_hi, 8);

            uint8x16_t result = vcombine_u8(res_lo, res_hi);

            vst1q_u8(dst + x, result);
        }

        for (; x < cols; ++x) {
            int idx = x * 3;

            uint8_t B = src[idx + 0];
            uint8_t G = src[idx + 1];
            uint8_t R = src[idx + 2];
            
            uint16_t sum = (uint16_t)R * w_r + (uint16_t)G * w_g + (uint16_t)B * w_b;
            dst[x] = (uint8_t)(sum >> 8);
        }
    }
}