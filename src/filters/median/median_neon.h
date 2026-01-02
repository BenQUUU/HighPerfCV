#pragma once

#include "../../../include/IFilter.h"

class MedianNEON : public IFilter {
public:
    explicit MedianNEON(int kernel_size);
    ~MedianNEON() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
};