#include "brightness_base.h"
#include <stdexcept>

BrightnessBase::BrightnessBase(float alpha, int beta)
    : alpha(alpha), beta(beta) {}

std::string BrightnessBase::get_name() const {
    return "Brightness/Contrast Filter (Base C++)";
}

void BrightnessBase::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.empty()) {
        return;
    }

    outputImage.create(inputImage.rows, inputImage.cols, inputImage.type());

    const int rows = inputImage.rows;
    const int cols = inputImage.cols * inputImage.channels();

    for (int row = 0; row < rows; ++row) {
        const uchar* ptr_in = inputImage.ptr<uchar>(row);
        uchar* ptr_out = outputImage.ptr<uchar>(row);

        for (int col = 0; col < cols; ++col) {
            uchar val = ptr_in[col];

            float result = alpha * val + beta;

            ptr_out[col] = cv::saturate_cast<uchar>(result);
        }
    }
}
