#pragma once

#include "IFilter.h"
#include "utils.h"
#include <memory>

class FilterFactory {
public:
    static std::unique_ptr<IFilter> create_filter(FilterType filterType, OptimizationMode mode);
};
