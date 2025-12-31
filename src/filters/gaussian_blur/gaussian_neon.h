#pragma once

#include "../../../include/IFilter.h"
#include <vector>

class GaussianNEON : public IFilter {
public:
    GaussianNEON(int kernel_size, float sigma);
    ~GaussianNEON() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    [[nodiscard]] std::string get_name() const override;

private:
    int kernel_size;
    float sigma;
    std::vector<float> kernel;

    void generateKernel();
};