#pragma once

#include "../../../include/IFilter.h"

class GrayscaleNEON : public IFilter {
public:
    GrayscaleNEON() = default;
    ~GrayscaleNEON() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;
};