#pragma once

#include <string>
#include <stdexcept>

enum class FilterType {
    GRAYSCALE,
    BRIGHTNESS_CONTRAST,
    GAUSSIAN_BLUR,
    MEDIAN
};

enum class OptimizationMode {
    BASE,
    OPENMP,
    AVX2,
    CUDA
};

inline FilterType string_to_filter(const std::string& str) {
    if (str == "GRAYSCALE") return FilterType::GRAYSCALE;
    if (str == "BRIGHTNESS_CONTRAST") return FilterType::BRIGHTNESS_CONTRAST;
    if (str == "GAUSSIAN_BLUR") return FilterType::GAUSSIAN_BLUR;
    if (str == "MEDIAN") return FilterType::MEDIAN;
    throw std::invalid_argument("Unknown filter type: " + str);
}

inline OptimizationMode string_to_mode(const std::string& str) {
    if (str == "BASE") return OptimizationMode::BASE;
    if (str == "OPENMP") return OptimizationMode::OPENMP;
    if (str == "AVX") return OptimizationMode::AVX2;
    if (str == "CUDA") return OptimizationMode::CUDA;
    throw std::invalid_argument("Unknown optimization mode: " + str);
}
