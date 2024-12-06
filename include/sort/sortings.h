#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include <future>
#include "../gpu/cl.hpp"
#include "../gpu/kernels.h"
#include <iostream>
#include "../gpu/utils.h"
#include "../utils/data_loader.h"
#include "../utils/graph_printer.h"
#include "../utils/performance_stats.h"
#include "bitonic_sort.h"
#include "../math/basic_math.h"
#include <execution>
#include "../math/calc_type.h"

#define EMPTY_VECTOR_MESSAGE "Data vector is empty"
#define EMPTY_DEVIATIONS_MESSAGE "Deviations vector is empty"
#define ALMOST_ZERO 1e-9

std::vector<double> sortData(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

template <typename ExecutionPolicy>
std::vector<double> sortDataWithPolicy(const std::vector<double>& data, ExecutionPolicy policy, PerformanceStats& stats);

std::vector<double> sortDataVectorized(const std::vector<double>& data, PerformanceStats& stat);

std::vector<double> sortDataOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);
