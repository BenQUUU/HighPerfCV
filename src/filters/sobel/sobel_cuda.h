#pragma once

#include "../../../include/IFilter.h"

class SobelCUDA : public IFilter {
public:
    SobelCUDA() = default;
    ~SobelCUDA() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    std::string get_name() const override;
};