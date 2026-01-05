#pragma once

#include "../../../include/IFilter.h"

class SobelBase : public IFilter {
public:
    SobelBase() = default;
    ~SobelBase() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;
};