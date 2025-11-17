#pragma once

#include "../../../include/IFilter.h"

class GrayscaleCUDA : public IFilter {
public:
    ~GrayscaleCUDA() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;

    std::string get_name() const override;
};

