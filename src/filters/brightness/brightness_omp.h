#pragma once

#include "../../../include/IFilter.h"

class BrightnessOpenMP : public IFilter {
public:
    BrightnessOpenMP(float alpha, int beta);
    ~BrightnessOpenMP() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;

    [[nodiscard]] std::string get_name() const override;

private:
    float alpha;
    int beta;
};
