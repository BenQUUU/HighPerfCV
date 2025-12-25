#pragma once

#include "../../../include/IFilter.h"

class MedianCUDA : public IFilter {
public:
    explicit MedianCUDA(int kernel_size);
    ~MedianCUDA() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
};