#include "brightness_neon.h"
#include <arm_neon.h>
#include <omp.h>
#include <cmath>

BrightnessNEON::BrightnessNEON(float alpha, int beta) : alpha(alpha), beta(beta) {}

std::string BrightnessNEON::get_name() const {
    return "Brightness & Contrast (ARM NEON)";
}

void BrightnessNEON::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;

    output.create(input.rows, input.cols, input.type());

    int rows = input.rows;
    int cols = input.cols * input.channels();

    if (input.isContinuous() && output.isContinuous()) {
        cols = rows * cols;
        rows = 1;
    }

    float32x4_t v_alpha = vdupq_n_f32(alpha);
    float32x4_t v_beta = vdupq_n_f32(static_cast<float>(beta));

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const uint8_t* ptr_in = input.ptr<uint8_t>(y);
        uint8_t* ptr_out = output.ptr<uint8_t>(y);

        int x = 0;
        
        for (; x <= cols - 16; x += 16) {
            uint8x16_t v_u8 = vld1q_u8(ptr_in + x);

            uint16x8_t v_u16_low = vmovl_u8(vget_low_u8(v_u8));
            uint16x8_t v_u16_high = vmovl_high_u8(v_u8);

            uint32x4_t v_u32_0 = vmovl_u16(vget_low_u16(v_u16_low));
            float32x4_t v_f32_0 = vcvtq_f32_u32(v_u32_0);
            
            uint32x4_t v_u32_1 = vmovl_u16(vget_high_u16(v_u16_low));
            float32x4_t v_f32_1 = vcvtq_f32_u32(v_u32_1);

            uint32x4_t v_u32_2 = vmovl_u16(vget_low_u16(v_u16_high));
            float32x4_t v_f32_2 = vcvtq_f32_u32(v_u32_2);

            uint32x4_t v_u32_3 = vmovl_u16(vget_high_u16(v_u16_high));
            float32x4_t v_f32_3 = vcvtq_f32_u32(v_u32_3);
            
            v_f32_0 = vmlaq_f32(v_beta, v_f32_0, v_alpha);
            v_f32_1 = vmlaq_f32(v_beta, v_f32_1, v_alpha);
            v_f32_2 = vmlaq_f32(v_beta, v_f32_2, v_alpha);
            v_f32_3 = vmlaq_f32(v_beta, v_f32_3, v_alpha);

            v_u32_0 = vcvtaq_u32_f32(v_f32_0);
            v_u32_1 = vcvtaq_u32_f32(v_f32_1);
            v_u32_2 = vcvtaq_u32_f32(v_f32_2);
            v_u32_3 = vcvtaq_u32_f32(v_f32_3);

            uint16x4_t v_res_u16_0 = vqmovn_u32(v_u32_0);
            uint16x4_t v_res_u16_1 = vqmovn_u32(v_u32_1);
            
            uint16x4_t v_res_u16_2 = vqmovn_u32(v_u32_2);
            uint16x4_t v_res_u16_3 = vqmovn_u32(v_u32_3);

            uint16x8_t v_res_u16_low = vcombine_u16(v_res_u16_0, v_res_u16_1);
            uint16x8_t v_res_u16_high = vcombine_u16(v_res_u16_2, v_res_u16_3);

            uint8x8_t v_res_u8_low = vqmovn_u16(v_res_u16_low);
            uint8x8_t v_res_u8_high = vqmovn_u16(v_res_u16_high);

            uint8x16_t v_result = vcombine_u8(v_res_u8_low, v_res_u8_high);

            vst1q_u8(ptr_out + x, v_result);
        }

        for (; x < cols; ++x) {
            float val = ptr_in[x] * alpha + beta;
            ptr_out[x] = cv::saturate_cast<uint8_t>(val);
        }
    }
}