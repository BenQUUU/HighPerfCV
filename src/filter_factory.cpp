#include <memory>
#include <stdexcept>
#include "../include/FilterFactory.h"

#include "filters/grayscale/grayscale_base.h"
#include "filters/grayscale/grayscale_omp.h"

#include "filters/brightness/brightness_base.h"
#include "filters/brightness/brightness_omp.h"

#include "filters/gaussian_blur/gaussian_base.h"
#include "filters/gaussian_blur/gaussian_omp.h"

#include "filters/median/median_base.h"
#include "filters/median/median_omp.h"

#ifdef USE_AVX2
#include "filters/gaussian_blur/gaussian_avx.h"
#include "filters/brightness/brightness_avx.h"
#include "filters/grayscale/grayscale_avx.h"
#endif

#ifdef USE_CUDA
#include "filters/brightness/brightness_cuda.h"
#include "filters/gaussian_blur/gaussian_cuda.h"
#include "filters/grayscale/grayscale_cuda.h"
#endif

std::unique_ptr<IFilter> FilterFactory::create_filter(FilterType filterType, OptimizationMode mode, const std::vector<std::string>& params) {
    switch (filterType) {
    case FilterType::GRAYSCALE:
        switch (mode) {
    case OptimizationMode::BASE:
            return std::make_unique<GrayscaleBase>();
    case OptimizationMode::OPENMP:
            return std::make_unique<GrayscaleOpenMP>();
#ifdef USE_AVX2
    case OptimizationMode::AVX2:
            return std::make_unique<GrayscaleAVX>();
#endif
#ifdef USE_CUDA
    case OptimizationMode::CUDA:
            return std::make_unique<GrayscaleCUDA>();
#endif
    default:
            throw std::invalid_argument("Unknown optimization mode for Grayscale filter");
        }
    case FilterType::BRIGHTNESS_CONTRAST: {
        float alpha = 1.3f;
        int beta = 20;

        try {
            if (params.size() >= 1) {
                alpha = std::stof(params[0]);
            }
            if (params.size() >= 2) {
                beta = std::stof(params[1]);
            }
        } catch (std::exception& e) {
            throw std::invalid_argument("Incorrect parameters for Brightness! Expected numbers. " + std::string(e.what()));
        }

        switch (mode) {
        case OptimizationMode::BASE:
            return std::make_unique<BrightnessBase>(alpha, beta);
        case OptimizationMode::OPENMP:
            return std::make_unique<BrightnessOpenMP>(alpha, beta);
#ifdef USE_AVX2
        case OptimizationMode::AVX2:
            return std::make_unique<BrightnessAVX>(alpha, beta);
#endif
#ifdef USE_CUDA
        case OptimizationMode::CUDA:
            return std::make_unique<BrightnessCUDA>(alpha, beta);
#endif
        default:
            throw std::invalid_argument("Unknown optimization mode for Brightness filter");
        }
    }

    case FilterType::GAUSSIAN_BLUR: {
        int k_size = 5;
        float sigma = 1.0f;

        try {
            if (params.size() >= 1)
                k_size = std::stoi(params[0]);
            if (params.size() >= 2)
                sigma = std::stof(params[1]);
        } catch (...) {
            throw std::invalid_argument("Incorrect parameters for Gaussian! Expected: int (size), float (sigma)");
        }

        switch (mode) {
        case OptimizationMode::BASE:
            return std::make_unique<GaussianBase>(k_size, sigma);
        case OptimizationMode::OPENMP:
            return std::make_unique<GaussianOpenMP>(k_size, sigma);
#ifdef USE_AVX2
        case OptimizationMode::AVX2:
            return std::make_unique<GaussianAVX>(k_size, sigma);
#endif
#ifdef USE_CUDA
        case OptimizationMode::CUDA:
            return std::make_unique<GaussianCUDA>(k_size, sigma);
#endif
        default:
            throw std::runtime_error("Unknown optimization mode for GaussianBlur");
        }
    }

    case FilterType::MEDIAN: {
        int k_size = 3;

        try {
            if (!params.empty()) k_size = std::stoi(params[0]);
        } catch (...) {
            throw std::invalid_argument("Median requires a parameter int (kernel_size).");
        }

        switch (mode) {
        case OptimizationMode::BASE:
            return std::make_unique<MedianBase>(k_size);
        case OptimizationMode::OPENMP:
            return std::make_unique<MedianOpenMP>(k_size);
        default:
            throw std::runtime_error("Unknown optimization mode for Median Filter");
        }
    }

        throw std::runtime_error("Unknown combination of filter type and optimization mode");
    }
}
