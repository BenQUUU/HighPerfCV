#include "../include/FilterFactory.h"
#include <memory>
#include <stdexcept>

#include "filters/grayscale/grayscale_base.h"
#include "filters/grayscale/grayscale_omp.h"
#include "filters/grayscale/grayscale_avx.h"

#ifdef USE_CUDA
#include "filters/grayscale/grayscale_cuda.h"
#endif

std::unique_ptr<IFilter> FilterFactory::create_filter(FilterType filterType, OptimizationMode mode) {
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
            }
            break;
    case FilterType::BRIGHTNESS_CONTRAST:
        // TODO: Zaimplementować, gdy klasy będą gotowe
        // Np. return std::make_unique<BrightnessBase>();
        break;

    case FilterType::GAUSSIAN_BLUR:
        // TODO: Zaimplementować, gdy klasy będą gotowe
        break;

    case FilterType::MEDIAN:
        // TODO: Zaimplementować, gdy klasy będą gotowe
        break;

    }

    throw std::runtime_error("Unknown combination of filter type and optimization mode");
}

