#pragma once

#include "../../../include/IFilter.h"

class BrightnessBase : public IFilter {
public:
    BrightnessBase(float alpha, int beta);
    ~BrightnessBase() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;

    [[nodiscard]] std::string get_name() const override;
private:
    float alpha;
    int beta;
};
