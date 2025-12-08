#include <memory>
#include <stdexcept>
#include "../include/FilterFactory.h"

#include "filters/grayscale/grayscale_avx.h"
#include "filters/grayscale/grayscale_base.h"
#include "filters/grayscale/grayscale_omp.h"

#include "filters/brightness/brightness_avx.h"
#include "filters/brightness/brightness_base.h"
#include "filters/brightness/brightness_omp.h"

#include "filters/gaussian_blur/gaussian_base.h"

#ifdef USE_CUDA
#include "filters/grayscale/grayscale_cuda.h"
#include "filters/brightness/brightness_cuda.h"
#endif

std::unique_ptr<IFilter> FilterFactory::create_filter(FilterType filterType, OptimizationMode mode, const std::vector<std::string>& params) {
    switch (filterType) {
    case FilterType::GRAYSCALE:
        switch (mode) {
        case OptimizationMode::BASE:
            return std::make_unique<GrayscaleBase>();
        case OptimizationMode::OPENMP:
            return std::make_unique<GrayscaleOpenMP>();
        case OptimizationMode::AVX2:
            return std::make_unique<GrayscaleAVX>();
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
        case OptimizationMode::AVX2:
            return std::make_unique<BrightnessAVX>(alpha, beta);
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
            if (params.size() >= 1) k_size = std::stoi(params[0]);
            if (params.size() >= 2) sigma = std::stof(params[1]);
        } catch (...) {
            throw std::invalid_argument("Bledne parametry dla Gaussian! Oczekiwano: int (rozmiar), float (sigma).");
        }

        switch (mode) {
        case OptimizationMode::BASE:
            return std::make_unique<GaussianBase>(k_size, sigma);
        default:
            throw std::runtime_error("Tryb niedostepny dla GaussianBlur");
        }
    }

    case FilterType::MEDIAN:
        // TODO: Zaimplementować, gdy klasy będą gotowe
        break;
    }

    throw std::runtime_error("Unknown combination of filter type and optimization mode");
}
