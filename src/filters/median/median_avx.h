#pragma once

#include "../../../include/IFilter.h"

class MedianAVX : public IFilter {
public:
    explicit MedianAVX(int kernel_size);
    ~MedianAVX() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
};
