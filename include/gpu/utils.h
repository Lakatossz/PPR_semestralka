#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "kernels.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include <future>
#include <iostream>
#include <execution>

cl::Platform getPlatform();

cl::Device getDevice(cl::Platform platform);

cl::Context getContext();

cl::Context prepareForGPU(cl::CommandQueue &queue, cl::Program &program);

cl::Buffer prepareDataBuffer(const cl::Context context, const std::vector<double>& data);

void sumMeanUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_partial_sum, size_t global_size, size_t local_size, int n);

void sumVarUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_variance, double mean, size_t global_size, size_t local_size, int n);

void sortDataUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_sorted, int n);

void calculateAbsoluteDeviationUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_deviations, double median, int n);