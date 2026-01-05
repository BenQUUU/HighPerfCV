#include "sobel_base.h"
#include <cmath>
#include <algorithm>

std::string SobelBase::get_name() const {
    return "Sobel Edge Detection (Base C++)";
}

void SobelBase::process(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) return;

    output.create(input.rows, input.cols, input.type());

    const int rows = input.rows;
    const int cols = input.cols;
    const int channels = input.channels();

    for (int y = 1; y < rows - 1; ++y) {
        const uchar* ptr_prev = input.ptr<uchar>(y - 1);
        const uchar* ptr_curr = input.ptr<uchar>(y);
        const uchar* ptr_next = input.ptr<uchar>(y + 1);
        
        uchar* ptr_out = output.ptr<uchar>(y);

        for (int x = 1; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                int idx_l = (x - 1) * channels + c;;
                int idx_c = (x) * channels + c;
                int idx_r = (x + 1) * channels + c;

                int gx = -1 * ptr_prev[idx_l] + 0 * ptr_prev[idx_c] + 1 * ptr_prev[idx_r]
                         -2 * ptr_curr[idx_l] + 0 * ptr_curr[idx_c] + 2 * ptr_curr[idx_r]
                         -1 * ptr_next[idx_l] + 0 * ptr_next[idx_c] + 1 * ptr_next[idx_r];

                int gy = -1 * ptr_prev[idx_l] - 2 * ptr_prev[idx_c] - 1 * ptr_prev[idx_r]
                         +0 * ptr_curr[idx_l] + 0 * ptr_curr[idx_c] + 0 * ptr_curr[idx_r]
                         +1 * ptr_next[idx_l] + 2 * ptr_next[idx_c] + 1 * ptr_next[idx_r];

                // G = sqrt(gx^2 + gy^2)
                float magnitude = std::sqrt(static_cast<float>(gx * gx + gy * gy));

                ptr_out[idx_c] = cv::saturate_cast<uchar>(magnitude);
            }
        }
    }

    memset(output.ptr<uchar>(0), 0, cols * channels);
    memset(output.ptr<uchar>(rows - 1), 0, cols * channels);

    for(int y=0; y<rows; ++y) {
        uchar* row = output.ptr<uchar>(y);
        for(int c=0; c<channels; ++c) {
            row[c] = 0;
            row[(cols-1)*channels + c] = 0;
        }
    }
}