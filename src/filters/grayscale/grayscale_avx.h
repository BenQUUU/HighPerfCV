#pragma once

#include "../../../include/IFilter.h"

class GrayscaleAVX : public IFilter {
public:
    ~GrayscaleAVX() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;

    [[nodiscard]] std::string get_name() const override;
};
