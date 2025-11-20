#pragma once

#include "IFilter.h"
#include "utils.h"
#include <memory>
#include <vector>

class FilterFactory {
public:
    static std::unique_ptr<IFilter> create_filter(FilterType filterType, OptimizationMode mode, const std::vector<std::string>& params = {});
};
