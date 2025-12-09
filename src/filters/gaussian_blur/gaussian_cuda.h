#pragma once

#include "../../../include/IFilter.h"
#include <vector>

class GaussianCUDA : public IFilter {
public:
    GaussianCUDA(int kernel_size, float sigma);
    ~GaussianCUDA() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
    float sigma;
    std::vector<float> kernel;
};
