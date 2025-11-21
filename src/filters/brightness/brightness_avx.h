#pragma once

#include "../../../include/IFilter.h"

class BrightnessAVX : public IFilter {
public:
    BrightnessAVX(float alpha, int beta);
    ~BrightnessAVX() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;

    [[nodiscard]] std::string get_name() const override;

private:
    float alpha;
    int beta;
};
