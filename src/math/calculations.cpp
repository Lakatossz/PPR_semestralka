#include "../../include/math/calculations.h"

double sumMean(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    stat.stopTimer();
    switch (calcType) {
        case Serial:
            return sumMeanWithPolicy(data, std::execution::seq, stat);
            break;
        case Vectorized:
            return sumMeanVectorized(data, stat);
            break;
        case MultiThreadNonVectorized:
            return sumMeanWithPolicy(data, std::execution::par, stat);
            break;
        case ParallelVectorized:
            return sumMeanWithPolicy(data, std::execution::par_unseq, stat);
            break;
        default:
            return sumMeanOnGPU(data, queue, program, buffer_data, stat);
            break;
    }
}

double sumVar(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{

    stat.stopTimer();

    switch (calcType) {
        case Serial:
            return sumVarWithPolicy(data, mean, std::execution::seq, stat);
            break;
        case Vectorized:
            return sumVarVectorized(data, mean, stat);
            break;
        case MultiThreadNonVectorized:
            return sumVarWithPolicy(data, mean, std::execution::par, stat);
            break;
        case ParallelVectorized:
            return sumVarWithPolicy(data, mean, std::execution::par_unseq, stat);
            break;
        default:
            return sumVarOnGPU(data, mean, queue, program, buffer_data, stat);
            break;
    }
}

std::vector<double> calculateAbsDev(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    switch (calcType) {
        case Serial:
            return calculateAbsDevWithPolicy(data, mean, std::execution::seq, stat);
            break;
        case Vectorized:
            return calculateAbsDevVectorized(data, mean, stat);
            break;
        case MultiThreadNonVectorized:
            return calculateAbsDevWithPolicy(data, mean, std::execution::par, stat);
            break;
        case ParallelVectorized:
            return calculateAbsDevWithPolicy(data, mean, std::execution::par_unseq, stat);
            break;
        default:
            return calculateAbsDevOnGPU(data, mean, queue, program, buffer_data, stat);
            break;
    }
}

double calculateCV(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    controlDateVector(data);

    // Size of input data.
    size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;

    stat.stopTimer();

    // First we need to get the mean of the input data.
    mean = calcMean(sumMean(data, calcType, queue, program, buffer_data, stat), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    stat.stopTimer();

    // Finalize variance calculation
    variance = calcVar(sumVar(data, mean, calcType, queue, program, buffer_data, stat), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    stat.stopTimer();

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double calculateMAD(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    controlDateVector(data);

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> sortedData = sortData(data, calcType, queue, program, buffer_data, stat);

    // Get the median from sorted input data.
    double median = getMedian(sortedData);

    stat.stopTimer();

    std::vector<double> deviations = calculateAbsDev(data, median, calcType, queue, program, buffer_data, stat);
    if (deviations.size() == 0) {
        throw std::invalid_argument(EMPTY_DEVIATIONS_MESSAGE);
    }

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

template <typename ExecutionPolicy>
double sumMeanWithPolicy(const std::vector<double>& data, const ExecutionPolicy policy, PerformanceStats& stat)
{
    controlDateVector(data);

    unsigned int num_threads = std::thread::hardware_concurrency();

    size_t num_chunks = num_threads;
    size_t chunk_size = data.size() / num_chunks;

    // Vector to store partial sums from each chunk
    std::vector<double> partial_sums(num_chunks, 0.0);

    // Lambda to compute sum of a chunk
    auto compute_chunk_sum = [&](size_t start_index, size_t end_index) {
        return std::reduce(data.begin() + start_index, data.begin() + end_index, 0.0);
    };

    // Compute partial sums in parallel
    std::for_each(policy, partial_sums.begin(), partial_sums.end(), [&](double& partial_sum) {
        size_t chunk_index = &partial_sum - &partial_sums[0]; // Get the index of the current chunk
        size_t start_index = chunk_index * chunk_size;
        size_t end_index = (chunk_index == num_chunks - 1) ? data.size() : start_index + chunk_size; 

        partial_sum = compute_chunk_sum(start_index, end_index);
    });

    // Sum up partial results (this can also be parallelized if necessary)
    double total_sum = std::reduce(partial_sums.begin(), partial_sums.end(), 0.0);

    stat.stopTimer();

    return total_sum;
}

template <typename ExecutionPolicy>
double sumVarWithPolicy(const std::vector<double>& data, const double mean, const ExecutionPolicy policy, PerformanceStats& stat)
{
    controlDateVector(data);

    unsigned int num_threads = std::thread::hardware_concurrency();
    size_t num_chunks = num_threads;
    size_t chunk_size = data.size() / num_chunks;

    // Handle edge case: fewer data points than threads
    if (chunk_size == 0) {
        num_chunks = data.size();
        chunk_size = 1;
    }

    // Vector to store partial variances
    std::vector<double> temp_variations(num_chunks, 0.0);

    // Lambda for calculating variance contribution of a single value
    auto addVariance = [mean](double value) -> double {
        return (value - mean) * (value - mean);
    };

    stat.stopTimer();

    // Compute variance in chunks
    std::for_each(policy, temp_variations.begin(), temp_variations.end(), [&](double& partial_variance) {
        size_t chunk_index = &partial_variance - &temp_variations[0]; // Get the index of the current chunk
        size_t start_index = chunk_index * chunk_size;
        size_t end_index = (chunk_index == num_chunks - 1) ? data.size() : start_index + chunk_size;

        for (size_t i = start_index; i < end_index; ++i) {
            partial_variance += addVariance(data[i]);
        }
    });

    stat.stopTimer();

    // Sum up partial variances to get the total variance
    double total_variance = std::reduce(temp_variations.begin(), temp_variations.end(), 0.0);

    stat.stopTimer();

    return total_variance;
}

double getVectorResult(const __m256d var_vec, PerformanceStats& stat) {

    stat.stopTimer();

    alignas(32) double temp[4];
    // Extract the sum from the AVX register and add it to result.
    _mm256_store_pd(temp, var_vec);

    // Calculate the rest of mean.
    return temp[0] + temp[1] + temp[2] + temp[3];
}

double sumMeanVectorized(const std::vector<double>& data, PerformanceStats& stat)
{
    controlDateVector(data);

    // Size of input data.
    size_t n = data.size();

    // Initialling of AVX register for accumulating sum.
    __m256d sum_vec = _mm256_setzero_pd(), val;

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for (size_t i = 0; i < n; i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        val = _mm256_loadu_pd(&data[i]);

        // Accumulate values into sum_vec.
        sum_vec = _mm256_add_pd(sum_vec, val);
    }

    stat.stopTimer();

    return getVectorResult(sum_vec, stat);
}

double sumVarVectorized(const std::vector<double>& data, const double mean, PerformanceStats& stat)
{
    controlDateVector(data);

    // Size of input data.
    size_t n = data.size();

    // Initialling of AVX register for accumulating variance.
    __m256d var_vec = _mm256_setzero_pd(), val, diff;

    stat.stopTimer();

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for (size_t i = 0; i < n; i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        val = _mm256_loadu_pd(&data[i]);

        // Compute the difference between each element in val vector and the value of mean.
        diff = _mm256_sub_pd(val, _mm256_set1_pd(mean));

        // Accumulate into var_vec the square of diff.
        var_vec = _mm256_add_pd(var_vec, _mm256_mul_pd(diff, diff));
    }

    stat.stopTimer();

    return getVectorResult(var_vec, stat);
}

double sumMeanOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    controlDateVector(data);

    const size_t n = data.size();
    const size_t local_size = 256;   // Work-group size (depends on your device)
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size; // Round up

    double mean = 0.0;

    try {
        cl::Context context = getContext();

        // Step 1: Allocate buffers
        cl::Buffer buffer_partial_sum(context, CL_MEM_WRITE_ONLY, sizeof(double) * (global_size / local_size));
        cl::Buffer buffer_mean(context, CL_MEM_WRITE_ONLY, sizeof(double));
        cl::Buffer buffer_variance(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

        sumMeanUsingGPUKernel(queue, program, buffer_data, buffer_partial_sum, global_size, local_size, n);

        // Step 4: Retrieve the partial sums from the device
        std::vector<double> partial_sums(global_size / local_size);
        queue.enqueueReadBuffer(buffer_partial_sum, CL_TRUE, 0, sizeof(double) * partial_sums.size(), partial_sums.data());

        // Sum the partial sums on the host to get the final result
        for (double partial_sum : partial_sums) {
            mean += partial_sum;
        }

        stat.stopTimer();

        return mean;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error (sumMeanOnGPU): " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0.0;
}

double sumVarOnGPU(const std::vector<double>& data, const double mean, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    controlDateVector(data);

    const size_t n = data.size();
    const size_t local_size = 256;   // Work-group size (depends on your device)
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size; // Round up

    double variance = 0.0f;

    try {
        cl::Context context = getContext();

        // Step 1: Allocate buffers
        cl::Buffer buffer_partial_sum(context, CL_MEM_WRITE_ONLY, sizeof(double) * (global_size / local_size));
        cl::Buffer buffer_mean(context, CL_MEM_WRITE_ONLY, sizeof(double));
        cl::Buffer buffer_variance(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

        stat.stopTimer();

        // Step 3: Calculate Variance
        sumVarUsingGPUKernel(queue, program, buffer_data, buffer_variance, mean, global_size, local_size, n);

        // Read partial results for variance
        std::vector<double> partial_vars(global_size / local_size);
        queue.enqueueReadBuffer(buffer_variance, CL_TRUE, 0, sizeof(double) * partial_vars.size(), partial_vars.data());

        stat.stopTimer();

        // Calculate final variance on host
        variance = std::accumulate(partial_vars.begin(), partial_vars.end(), 0.0);

        stat.stopTimer();

        return variance;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error (sumVarOnGPU): " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0.0;
}

std::vector<double> calculateAbsDevVectorized(const std::vector<double>& data, const double median, PerformanceStats& stat) 
{ 
    controlDateVector(data);

    size_t i = 0, n = data.size();

    __m256d sign_mask, vec_data, vec_median, vec_dev;

    std::vector<double> deviations(n);

    // Mask for setting abs value in vector (IEEE 754 has the sign bit set).
    sign_mask = _mm256_set1_pd(-0.0);

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for(; i + 4 <= data.size(); i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        vec_data = _mm256_loadu_pd(&data[i]);

        // Lets set all four lanes of a 256-bit AVX register vec_median to 
        // the same double-precision floating-point value.
        vec_median = _mm256_set1_pd(median);

        // Calculate the substraction of all values from median.
        vec_dev = _mm256_sub_pd(vec_data, vec_median);

        // Absolute value of the substraction.
        vec_dev = _mm256_andnot_pd(sign_mask, vec_dev);

        // Accumulate values into deviations vector.
        _mm256_storeu_pd(&deviations[i], vec_dev);
    }

    // This loop processes the data vector four doubles at a time (since AVX2 can operate on four doubles at once).
    for(; i < data.size(); ++i) {
        deviations[i] = getAbs(data[i], median);
    }

    stat.stopTimer();

    return sortDataVectorized(deviations, stat);
}

template <typename ExecutionPolicy>
std::vector<double> calculateAbsDevWithPolicy(const std::vector<double>& data, const double median, const ExecutionPolicy policy, PerformanceStats& stat) 
{
    controlDateVector(data);
    
    std::vector<double> deviations(data.size());

    std::transform(policy, data.begin(), data.end(), deviations.begin(), [&](const double& value) {
        return getAbs(value, median);
        });

    stat.stopTimer();

    return sortDataVectorized(deviations, stat);
}

std::vector<double> calculateAbsDevOnGPU(const std::vector<double>& data, const double median, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat) 
{   
    controlDateVector(data);

    const size_t n = data.size();

    std::vector<double> deviations(n);

    try {
        cl::Context context = getContext();

        cl::Buffer buffer_deviations(context, CL_MEM_WRITE_ONLY, n * sizeof(double));
        calculateAbsoluteDeviationUsingGPUKernel(queue, program, buffer_data, buffer_deviations, median, n);
        queue.enqueueReadBuffer(buffer_deviations, CL_TRUE, 0, n * sizeof(double), deviations.data());

        stat.stopTimer();

        return sortDataVectorized(deviations, stat);
        //return sortDataOnGPU(deviations, queue, program, buffer_data);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error (calculateAbsDevOnGPU): " << err.what() << "(" << err.err() << ")" << std::endl;
        return {};
    }
}

void controlDateVector(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }
}
