#pragma once

#include <cmath>  // For std::abs
#include <numeric>
#include <vector>
#include <algorithm>

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

void addVariance(double& variance, const double value, const double mean);

double getAbs(const double value, const double median);