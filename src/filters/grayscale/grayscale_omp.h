#pragma once

#include "../../../include/IFilter.h"

class GrayscaleOpenMP : public IFilter {
public:
    ~GrayscaleOpenMP() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;

    [[nodiscard]] std::string get_name() const override;
};