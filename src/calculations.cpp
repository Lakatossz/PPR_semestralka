#include "../include/calculations.h"

double sumMean(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
{
    switch (calcType) {
        case Serial:
            return sumMeanWithPolicy(data, std::execution::seq);
            break;
        case Vectorized:
            return sumMeanVectorized(data);
            break;
        case MultiThreadNonVectorized:
            return sumMeanWithPolicy(data, std::execution::par);
            break;
        case ParallelVectorized:
            return sumMeanWithPolicy(data, std::execution::par_unseq);
            break;
        default:
            return sumMeanOnGPU(data, queue, program, buffer_data);
            break;
    }
}

double sumVar(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
{
    switch (calcType) {
        case Serial:
            return sumVarWithPolicy(data, mean, std::execution::seq);
            break;
        case Vectorized:
            return sumVarVectorized(data, mean);
            break;
        case MultiThreadNonVectorized:
            return sumVarWithPolicy(data, mean, std::execution::par);
            break;
        case ParallelVectorized:
            return sumVarWithPolicy(data, mean, std::execution::par_unseq);
            break;
        default:
            return sumVarOnGPU(data, mean, queue, program, buffer_data);
            break;
    }
}

std::vector<double> calculateAbsDev(const std::vector<double>& data, const double mean, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
{
    switch (calcType) {
        case Serial:
            return calculateAbsDevWithPolicy(data, mean, std::execution::seq);
            break;
        case Vectorized:
            return calculateAbsDevVectorized(data, mean);
            break;
        case MultiThreadNonVectorized:
            return calculateAbsDevWithPolicy(data, mean, std::execution::par);
            break;
        case ParallelVectorized:
            return calculateAbsDevWithPolicy(data, mean, std::execution::par_unseq);
            break;
        default:
            return calculateAbsDevOnGPU(data, mean, queue, program, buffer_data);
            break;
    }
}

double calculateCV(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
{
    controlDateVector(data);

    // Size of input data.
    size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;

    // First we need to get the mean of the input data.
    mean = calcMean(sumMean(data, calcType, queue, program, buffer_data), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Finalize variance calculation
    variance = calcVar(sumVar(data, mean, calcType, queue, program, buffer_data), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double calculateMAD(const std::vector<double>& data,const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
{
    controlDateVector(data);

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> sortedData = sortData(data, calcType, queue, program, buffer_data);

    // Get the median from sorted input data.
    double median = getMedian(sortedData);

    std::vector<double> deviations = calculateAbsDev(data, median, calcType, queue, program, buffer_data);
    if (deviations.size() == 0) {
        throw std::invalid_argument(EMPTY_DEVIATIONS_MESSAGE);
    }

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

template <typename ExecutionPolicy>
double sumMeanWithPolicy(const std::vector<double>& data, const ExecutionPolicy policy)
{
    controlDateVector(data);
    return std::reduce(policy, data.begin(), data.end(), 0.0);
}

template <typename ExecutionPolicy>
double sumVarWithPolicy(const std::vector<double>& data, const double mean, const ExecutionPolicy policy)
{
    controlDateVector(data);
    double variance = 0.0;

    // Than we need to get the variance from calculated mean and each value.
    std::for_each(policy, data.begin(), data.end(), [&](const double& value) {
        addVariance(variance, value, mean);
    });

    return variance;
}


double getVectorResult(const __m256d var_vec) {
    // Extract the sum from the AVX register and add it to result.
    alignas(32) double temp[4];
    // Extract the sum from the AVX register and add it to result.
    _mm256_store_pd(temp, var_vec);

    // Calculate the rest of mean.
    return temp[0] + temp[1] + temp[2] + temp[3];
}

double sumMeanVectorized(const std::vector<double>& data)
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

    return getVectorResult(sum_vec);
}

double sumVarVectorized(const std::vector<double>& data, const double mean)
{
    controlDateVector(data);

    // Size of input data.
    size_t n = data.size();

    // Initialling of AVX register for accumulating variance.
    __m256d var_vec = _mm256_setzero_pd(), val, diff;

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for (size_t i = 0; i < n; i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        val = _mm256_loadu_pd(&data[i]);

        // Compute the difference between each element in val vector and the value of mean.
        diff = _mm256_sub_pd(val, _mm256_set1_pd(mean));

        // Accumulate into var_vec the square of diff.
        var_vec = _mm256_add_pd(var_vec, _mm256_mul_pd(diff, diff));
    }

    return getVectorResult(var_vec);
}

double sumMeanOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
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

        return mean;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error (sumMeanOnGPU): " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0.0;
}

double sumVarOnGPU(const std::vector<double>& data, const double mean, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data)
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

        // Step 3: Calculate Variance
        sumVarUsingGPUKernel(queue, program, buffer_data, buffer_variance, mean, global_size, local_size, n);

        // Read partial results for variance
        std::vector<double> partial_vars(global_size / local_size);
        queue.enqueueReadBuffer(buffer_variance, CL_TRUE, 0, sizeof(double) * partial_vars.size(), partial_vars.data());

        // Calculate final variance on host
        variance = std::accumulate(partial_vars.begin(), partial_vars.end(), 0.0);

        return variance;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error (sumVarOnGPU): " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0.0;
}

std::vector<double> calculateAbsDevVectorized(const std::vector<double>& data, const double median) 
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

    return sortDataVectorized(deviations);
}

template <typename ExecutionPolicy>
std::vector<double> calculateAbsDevWithPolicy(const std::vector<double>& data, const double median, const ExecutionPolicy policy) 
{
    controlDateVector(data);
    
    std::vector<double> deviations(data.size());
    
    std::transform(policy, data.begin(), data.end(), deviations.begin(), [&](const double& value) {
        return getAbs(value, median);
        });

    return sortDataParallelVectorized(deviations);
}

std::vector<double> calculateAbsDevOnGPU(const std::vector<double>& data, const double median, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data) 
{   
    controlDateVector(data);

    const size_t n = data.size();

    std::vector<double> deviations(n);

    try {
        cl::Context context = getContext();

        cl::Buffer buffer_deviations(context, CL_MEM_WRITE_ONLY, n * sizeof(double));
        calculateAbsoluteDeviationUsingGPUKernel(queue, program, buffer_data, buffer_deviations, median, n);
        queue.enqueueReadBuffer(buffer_deviations, CL_TRUE, 0, n * sizeof(double), deviations.data());

        return sortDataParallelVectorized(deviations);
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
