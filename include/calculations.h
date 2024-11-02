#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>  // For std::abs
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
#include <execution>

#define EMPTY_VECTOR_MESSAGE "Data vector is empty"
#define ALMOST_ZERO 1e-9

enum CalcType {
    Serial,
    Vectorized,
    MultiThreadNonVectorized,
    ParallelVectorized,
    OnGPU
};

std::string calcTypeToString(CalcType calcType);

/**
 *  Returns Mean from sum of array and its size.
 */
double calcMean(const double sum, const size_t size);

/**
 *  Returns Variance from sum of square root of difference between mean and all values from data and size of data.
 */
double calcVar(const double sumOfDifferences, const size_t size);

/**
 *  Returns Coefficient of Variation from Variance and Standard Deviation.
 */
double calcCV(const double variance, const double stddev);

/**
 * Returns Median of a vector.
 */
double getMedian(const std::vector<double>& data);

double calculateCV(const std::vector<double>& data,const CalcType calcType);

/**
 *  Returns Coefficient of Variation of vector of doubles calculated serially on CPU.
 */
double calculateCVSerial(const std::vector<double>& data);

/**
 *  Returns Coefficient of Variation of vector of doubles calculated vectorial on CPU.
 */
double calculateCVVectorized(const std::vector<double>& data);

double calculateCVMultiThreadNonVectorized(const std::vector<double>& data);

double calculateCVParallelVectorized(const std::vector<double>& data);

// Broken
double calculateCVOnGPU(std::vector<double> data);

double calculateMAD(const std::vector<double>& data,const CalcType calcType);

double calculateMADSerial(const std::vector<double>& data);

double calculateMADVectorized(const std::vector<double>& data);

double calculateMADMultiThreadNonVectorized(const std::vector<double>& data);

double calculateMADParallelVectorized(const std::vector<double>& data);

double calculateMADOnGPU(const std::vector<double>& data);
