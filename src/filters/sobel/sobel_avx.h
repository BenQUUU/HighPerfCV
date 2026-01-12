#pragma once

#include "../../../include/IFilter.h"

class SobelAVX : public IFilter {
public:
    SobelAVX() = default;
    ~SobelAVX() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    std::string get_name() const override;
};
