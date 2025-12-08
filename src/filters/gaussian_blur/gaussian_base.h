#pragma once

#include "../../../include/IFilter.h"

class GaussianBase : public IFilter {
public:
    GaussianBase(int kernel_size, float sigma);
    ~GaussianBase() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
    float sigma;

    std::vector<float> kernel;

    void generateKernel();
};