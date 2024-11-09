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
#include "cl.hpp"
#include "kernels.h"
#include <iostream>
#include "utils.h"
#include "data_loader.h"
#include "graph_printer.h"
#include "sort.h"
#include "basic_math.h"
#include <execution>
#include "calc_type.h"

#define EMPTY_VECTOR_MESSAGE "Data vector is empty"
#define EMPTY_DEVIATIONS_MESSAGE "Deviations vector is empty"
#define ALMOST_ZERO 1e-9

std::vector<double> sortData(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

template <typename ExecutionPolicy>
std::vector<double> sortDataWithPolicy(const std::vector<double>& data, const ExecutionPolicy policy);

std::vector<double> sortDataVectorized(const std::vector<double>& data);

std::vector<double> sortDataOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);
