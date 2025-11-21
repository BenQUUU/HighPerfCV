#pragma once

#include "../../../include/IFilter.h"

class BrightnessCUDA : public IFilter {
public:
    BrightnessCUDA(float alpha, int beta);
    ~BrightnessCUDA() override = default;

    void process(const cv::Mat& inputImage, cv::Mat& outputImage) override;

    [[nodiscard]] std::string get_name() const override;

private:
    float alpha;
    int beta;
};
