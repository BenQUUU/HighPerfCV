#pragma once

#include "../../../include/IFilter.h"
#include <vector>

class GaussianOpenMP : public IFilter {
public:
    GaussianOpenMP(int kernel_size, float sigma);
    ~GaussianOpenMP() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
    float sigma;
    std::vector<float> kernel;

    void generateKernel();
};
