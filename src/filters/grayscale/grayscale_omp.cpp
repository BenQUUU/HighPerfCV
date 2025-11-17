#include "grayscale_omp.h"
#include <stdexcept>
#include <omp.h>

std::string GrayscaleOpenMP::get_name() const {
    return "Grayscale (OpenMP)";
}

void GrayscaleOpenMP::process(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.channels() != 3) {
        throw std::invalid_argument("The input image for GrayscaleOpenMP must be 3-channel (BGR)");
    }

    outputImage.create(inputImage.rows, inputImage.cols, CV_8UC1);

    #pragma omp parallel for
    for (int row = 0; row < inputImage.rows; ++row) {
        const cv::Vec3b* ptr_in = inputImage.ptr<cv::Vec3b>(row);
        uchar* ptr_out = outputImage.ptr<uchar>(row);

        for (int col = 0; col < inputImage.cols; ++col) {
            const cv::Vec3b& pixel_in = ptr_in[col];

            float gray_val = 0.114f * pixel_in[0] +
                             0.587f * pixel_in[1] +
                             0.299f * pixel_in[2];

            ptr_out[col] = static_cast<uchar>(gray_val);
        }
    }
}
