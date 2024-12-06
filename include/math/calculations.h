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
#include "basic_math.h"
#include <execution>
#include "calc_type.h"
#include "../sort/sortings.h"
#include <tbb/tbb.h>

#define EMPTY_VECTOR_MESSAGE "Data vector is empty"
#define EMPTY_DEVIATIONS_MESSAGE "Deviations vector is empty"
#define ALMOST_ZERO 1e-9

double sumMean(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

double sumVar(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

std::vector<double> calculateAbsDev(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

double calculateCV(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

double calculateMAD(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

template <typename ExecutionPolicy>
double sumMeanWithPolicy(const std::vector<double>& data, const ExecutionPolicy policy, PerformanceStats& stat);

template <typename ExecutionPolicy>
double sumVarWithPolicy(const std::vector<double>& data, const double mean, const ExecutionPolicy policy, PerformanceStats& stat);

double getVectorResult(const __m256d var_vec, PerformanceStats& stat);

double sumMeanVectorized(const std::vector<double>& data, PerformanceStats& stat);

double sumVarVectorized(const std::vector<double>& data, const double mean, PerformanceStats& stat);

double sumMeanOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

double sumVarOnGPU(const std::vector<double>& data, const double mean, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

template <typename ExecutionPolicy>
std::vector<double> calculateAbsDevWithPolicy(const std::vector<double>& data, const double median, const ExecutionPolicy policy, PerformanceStats& stat);

std::vector<double> calculateAbsDevVectorized(const std::vector<double>& data, const double median, PerformanceStats& stat);

std::vector<double> calculateAbsDevOnGPU(const std::vector<double>& data, const double median, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat);

void controlDateVector(const std::vector<double>& data);
