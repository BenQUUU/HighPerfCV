#pragma once

#include "../../../include/IFilter.h"

class MedianOpenMP : public IFilter {
public:
    explicit MedianOpenMP(int kernel_size);
    ~MedianOpenMP() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
};
