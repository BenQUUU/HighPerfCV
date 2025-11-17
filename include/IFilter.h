#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class IFilter {
public:
    virtual ~IFilter() = default;

    virtual void process(const cv::Mat& inputImage, cv::Mat& outputImage) = 0;

    [[nodiscard]] virtual std::string get_name() const = 0;
};
