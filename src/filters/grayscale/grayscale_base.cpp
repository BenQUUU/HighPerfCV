#include "grayscale_base.h"
#include <stdexcept>

std::string GrayscaleBase::get_name() const {
    return "Grayscale (Base C++)";
}

void GrayscaleBase::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.channels() != 3) {
        throw std::invalid_argument("Input image must have 3 channels");
    }

    outputImage.create(inputImage.rows, inputImage.cols, CV_8UC1);

    for (size_t row = 0; row < inputImage.rows; ++row) {
        const cv::Vec3b* ptr_in = inputImage.ptr<cv::Vec3b>(row);
        uchar* ptr_out = outputImage.ptr<uchar>(row);

        for (size_t col = 0; col < inputImage.cols; ++col) {
            const cv::Vec3b& pixel_in = ptr_in[col];
            uchar b = pixel_in[0];
            uchar g = pixel_in[1];
            uchar r = pixel_in[2];

            float gray_value = 0.114f * b + 0.587f * g + 0.299f * r;
            ptr_out[col] = static_cast<uchar>(gray_value);
        }
    }
}


