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
#include "sortings.h"
#include <execution>

#define EMPTY_VECTOR_MESSAGE "Data vector is empty"
#define EMPTY_DEVIATIONS_MESSAGE "Deviations vector is empty"
#define ALMOST_ZERO 1e-9

double sumMean(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

double sumVar(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

std::vector<double> calculateAbsDev(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

double calculateCV(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

double calculateMAD(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

template <typename ExecutionPolicy>
double sumMeanWithPolicy(const std::vector<double>& data, const ExecutionPolicy policy);

template <typename ExecutionPolicy>
double sumVarWithPolicy(const std::vector<double>& data, const double mean, const ExecutionPolicy policy);

double getVectorResult(const __m256d var_vec);

double sumMeanVectorized(const std::vector<double>& data);

double sumVarVectorized(const std::vector<double>& data, const double mean);

double sumMeanOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

double sumVarOnGPU(const std::vector<double>& data, const double mean, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

template <typename ExecutionPolicy>
std::vector<double> calculateAbsDevWithPolicy(const std::vector<double>& data, const double median, const ExecutionPolicy policy);

std::vector<double> calculateAbsDevVectorized(const std::vector<double>& data, const double median);

std::vector<double> calculateAbsDevOnGPU(const std::vector<double>& data, const double median, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data);

void controlDateVector(const std::vector<double>& data);
