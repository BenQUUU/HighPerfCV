#pragma once

#include "../../../include/IFilter.h"

class SobelOpenMP : public IFilter {
public:
    SobelOpenMP() = default;
    ~SobelOpenMP() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;
};