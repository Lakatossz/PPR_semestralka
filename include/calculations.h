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
#define EMPTY_DEVIATIONS_MESSAGE "Deviations vector is empty"
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

double sumMean(const std::vector<double>& data, const CalcType calcType);

double sumVar(const std::vector<double>& data, const double mean, const CalcType calcType);

double calculateCV(const std::vector<double>& data,const CalcType calcType);

double sumMeanSerial(const std::vector<double>& data);

double sumVarSerial(const std::vector<double>& data, const double mean);

/**
 *  Returns Coefficient of Variation of vector of doubles calculated serially on CPU.
 */
double calculateCVSerial(const std::vector<double>& data);

double sumMeanVectorized(const std::vector<double>& data);

double sumVarVectorized(const std::vector<double>& data, const double mean);

/**
 *  Returns Coefficient of Variation of vector of doubles calculated vectorial on CPU.
 */
double calculateCVVectorized(const std::vector<double>& data);

double sumMeanMultiThreadNonVectorized(const std::vector<double>& data);

double sumVarMultiThreadNonVectorized(const std::vector<double>& data, const double mean);

double calculateCVMultiThreadNonVectorized(const std::vector<double>& data);

double sumMeanParallelVectorized(const std::vector<double>& data);

double sumVarParallelVectorized(const std::vector<double>& data, const double mean);

double calculateCVParallelVectorized(const std::vector<double>& data);

double sumMeannGPU(const std::vector<double>& data);

double sumVarnGPU(const std::vector<double>& data, const double mean);

double calculateCVOnGPU(std::vector<double> data);

double calculateMAD(const std::vector<double>& data,const CalcType calcType);

std::vector<double> sortDataSerial(const std::vector<double>& data);

std::vector<double> calculateAbsDevSerial(const std::vector<double>& data, const double median);

double calculateMADSerial(const std::vector<double>& data);

std::vector<double> sortDataVectorized(const std::vector<double>& data);

std::vector<double> calculateAbsDevVectorized(const std::vector<double>& data, const double median);

double calculateMADVectorized(const std::vector<double>& data);

std::vector<double> sortDataMultiThreadNonVectorized(const std::vector<double>& data);

std::vector<double> calculateAbsDevMultiThreadNonVectorized(const std::vector<double>& data, const double median);

double calculateMADMultiThreadNonVectorized(const std::vector<double>& data);

std::vector<double> sortDataParallelVectorized(const std::vector<double>& data);

std::vector<double> calculateAbsDevParallelVectorized(const std::vector<double>& data, const double median);

double calculateMADParallelVectorized(const std::vector<double>& data);

std::vector<double> sortDataOnGPU(const std::vector<double>& data);

std::vector<double> calculateAbsDevOnGPU(const std::vector<double>& data, const double median);

double calculateMADOnGPU(const std::vector<double>& data);
