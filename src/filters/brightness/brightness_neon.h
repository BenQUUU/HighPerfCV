#pragma once

#include "../../../include/IFilter.h"

class BrightnessNEON : public IFilter {
public:
    BrightnessNEON(float alpha, int beta);
    ~BrightnessNEON() override = default;

    void process(const cv::Mat& input, cv::Mat& output) override;
    
    [[nodiscard]] std::string get_name() const override;

private:
    float alpha;
    int beta;
};