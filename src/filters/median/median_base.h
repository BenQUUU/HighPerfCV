#pragma once

#include "../../../include/IFilter.h"

class MedianBase : public IFilter {
public:
    explicit MedianBase(int kernel_size);
    ~MedianBase() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
};
