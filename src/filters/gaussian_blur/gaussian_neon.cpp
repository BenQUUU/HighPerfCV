#include "gaussian_neon.h"
#include <cmath>
#include <algorithm>
#include <arm_neon.h> 
#include <omp.h>      
#include <stdexcept>

GaussianNEON::GaussianNEON(int kernel_size, float sigma)
    : kernel_size(kernel_size), sigma(sigma) 
{
    if (kernel_size % 2 == 0) throw std::invalid_argument("Kernel size must be odd");
    generateKernel();
}

std::string GaussianNEON::get_name() const {
    return "Gaussian Blur (Separable NEON + OpenMP)";
}

void GaussianNEON::generateKernel() {
    int r = kernel_size / 2;
    kernel.resize(kernel_size);
    float sum = 0.0f;
    float sigma2 = 2.0f * sigma * sigma;

    for (int x = -r; x <= r; ++x) {
        float val = std::exp(-(x * x) / sigma2);
        kernel[x + r] = val;
        sum += val;
    }

    for (float& k : kernel) {
        k /= sum;
    }
}

void GaussianNEON::process(const cv::Mat& input, cv::Mat& output) {
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

    const int total_floats = cols * 3;

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        uchar* ptr_out = output.ptr<uchar>(y);

        int x = 0;
        for (; x <= total_floats - 4; x += 4) {
            float32x4_t v_sum = vdupq_n_f32(0.0f);

            for (int k = -r; k <= r; ++k) {
                int ny = std::min(std::max(y + k, 0), rows - 1);
                
                float32x4_t v_weight = vdupq_n_f32(kernel[k + r]);

                const float* ptr_temp_row = temp.ptr<float>(ny);
                float32x4_t v_val = vld1q_f32(ptr_temp_row + x);

                v_sum = vmlaq_f32(v_sum, v_val, v_weight);
            }

            uint32x4_t v_res_u32 = vcvtaq_u32_f32(v_sum);

            uint16x4_t v_res_u16 = vqmovn_u32(v_res_u32);

            uint8x8_t v_res_u8 = vqmovn_u16(vcombine_u16(v_res_u16, v_res_u16));

            vst1_lane_u32(reinterpret_cast<uint32_t*>(ptr_out + x), 
                          reinterpret_cast<uint32x2_t>(v_res_u8), 0);
        }

        for (; x < total_floats; ++x) {
            float sum = 0.0f;
            for (int k = -r; k <= r; ++k) {
                int ny = std::min(std::max(y + k, 0), rows - 1);
                const float* ptr_temp_row = temp.ptr<float>(ny);
                sum += ptr_temp_row[x] * kernel[k + r];
            }
            ptr_out[x] = cv::saturate_cast<uchar>(sum);
        }
    }
}