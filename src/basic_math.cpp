#include "../include/basic_math.h"

double calcMean(const double sum, const size_t size)
{
    return sum / size;
}

double calcVar(const double sumOfDifferences, const size_t size)
{
    return sumOfDifferences / size;
}

double calcCV(const double mean, const double stddev)
{
    return (stddev / mean) * 100;
}

double getMedian(const std::vector<double>& data) {
    size_t size = data.size();
    return size % 2 == 0 ? (data[size / 2 - 1] + data[size / 2]) / 2.0 : data[size / 2];
}

void addVariance(double& variance, const double value, const double mean) {
    double diff = value - mean;
    // Accumulate the squared differences
    variance = variance + (diff * diff); 
}

double getAbs(const double value, const double median) {
    return std::fabs(value - median);
}