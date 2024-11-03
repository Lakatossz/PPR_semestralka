#include "../include/calculations.h"

std::string calcTypeToString(CalcType calcType)
{
    switch(calcType) {
        case Serial:
            return "Serial";
            break;
        case Vectorized:
            return "Vectorized";
            break;
        case MultiThreadNonVectorized:
            return "MultiThreadNonVectorized";
            break;
        case ParallelVectorized:
            return "ParallelVectorized";
            break;
        case OnGPU:
            return "OnGPU";
            break;
        default:
            return "None";
            break;
    }
}

double calcMean(const double sum, const size_t size)
{
    // Calculate the mean.
    return sum / size;
}

double calcVar(const double sumOfDifferences, const size_t size)
{
    // Calculate the variance.
    return sumOfDifferences / size;
}

double calcCV(const double mean, const double stddev)
{
    // Return final Coefficient of Variation value in percentage.
    return (stddev / mean) * 100;
}

double getMedian(const std::vector<double>& data) {
    size_t size = data.size();

    // Return value in the middle of vector.
    return size % 2 == 0 ? (data[size / 2 - 1] + data[size / 2]) / 2.0 : data[size / 2];
}

double sumMean(const std::vector<double>& data, const CalcType calcType)
{
    switch (calcType) {
        case Serial:
            return sumMeanSerial(data);
            break;
        case Vectorized:
            return sumMeanVectorized(data);
            break;
        case MultiThreadNonVectorized:
            return sumMeanMultiThreadNonVectorized(data);
            break;
        case ParallelVectorized:
            return sumMeanParallelVectorized(data);
            break;
        default:
            return sumMeannGPU(data);
            break;
    }
}

double sumVar(const std::vector<double>& data, const double mean, const CalcType calcType)
{
    switch (calcType) {
        case Serial:
            return sumVarSerial(data, mean);
            break;
        case Vectorized:
            return sumVarVectorized(data, mean);
            break;
        case MultiThreadNonVectorized:
            return sumVarMultiThreadNonVectorized(data, mean);
            break;
        case ParallelVectorized:
            return sumVarParallelVectorized(data, mean);
            break;
        default:
            return sumVarnGPU(data, mean);
            break;
    }
}

double calculateCV(const std::vector<double>& data,const CalcType calcType)
{
    switch (calcType) {
        case Serial:
            std::cout << "Provedu výpočet CV sériově." << std::endl;
            return calculateCVSerial(data);
            break;
        case Vectorized:
            std::cout << "Provedu výpočet CV vektorizovaně." << std::endl;
            return calculateCVVectorized(data);
            break;
        case MultiThreadNonVectorized:
            std::cout << "Provedu výpočet CV vícevláknově, nevektorizovaně." << std::endl;
            return calculateCVMultiThreadNonVectorized(data);
            break;
        case ParallelVectorized:
            std::cout << "Provedu výpočet CV vícevláknově, vektorizovaně." << std::endl;
            return calculateCVParallelVectorized(data);
            break;
        default:
            std::cout << "Provedu výpočet CV na GPU." << std::endl;
            return calculateCVOnGPU(data);
            break;
    }
}

double sumMeanSerial(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    return std::accumulate(data.begin(), data.end(), 0.0);
}

double sumVarSerial(const std::vector<double>& data, const double mean)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    double variance = 0.0;

    // Than we need to get the variance from calculated mean and each value.
    for (const auto& value : data) {
        double diff = value - mean;

        // Accumulate the squared differences
        variance += (diff) * (diff);
    }

    return variance;
}

double calculateCVSerial(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;

    // First we need to get the mean of the input data.
    mean = calcMean(sumMeanSerial(data), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Finalize variance calculation
    variance = calcVar(sumVarSerial(data, mean), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double sumMeanVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    size_t n = data.size();

    __m256d sum_vec, var_vec, val, diff;

    // Initialling of AVX register for accumulating sum.
    sum_vec = _mm256_setzero_pd();

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for (size_t i = 0; i < n; i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        val = _mm256_loadu_pd(&data[i]);

        // Accumulate values into sum_vec.
        sum_vec = _mm256_add_pd(sum_vec, val);
    }

    // Extract the sum from the AVX register and add it to result.
    alignas(32) double temp[4];
    _mm256_store_pd(temp, sum_vec);

    // Calculate the rest of mean.
    return (temp[0] + temp[1] + temp[2] + temp[3]);
}

double sumVarVectorized(const std::vector<double>& data, const double mean)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    size_t n = data.size();

    __m256d sum_vec, var_vec, val, diff;

    // Initialling of AVX register for accumulating variance.
    var_vec = _mm256_setzero_pd();

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for (size_t i = 0; i < n; i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        val = _mm256_loadu_pd(&data[i]);

        // Compute the difference between each element in val vector and the value of mean.
        diff = _mm256_sub_pd(val, _mm256_set1_pd(mean));

        // Accumulate into var_vec the square of diff.
        var_vec = _mm256_add_pd(var_vec, _mm256_mul_pd(diff, diff));
    }

    // Extract the sum from the AVX register and add it to result.
    alignas(32) double temp[4];
    // Extract the sum from the AVX register and add it to result.
    _mm256_store_pd(temp, var_vec);

    // Calculate the rest of mean.
    return temp[0] + temp[1] + temp[2] + temp[3];
}

double calculateCVVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;

    // Calculate the rest of mean.
    mean = calcMean(sumMeanVectorized(data), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Calculate the rest of mean.
    variance = calcVar(sumVarVectorized(data, mean), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double sumMeanMultiThreadNonVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }
    return std::reduce(std::execution::par, data.begin(), data.end(), 0.0);
}

double sumVarMultiThreadNonVectorized(const std::vector<double>& data, const double mean)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    double variance = 0.0;

    // Than we need to get the variance from calculated mean and each value.
    std::for_each(std::execution::par, data.begin(), data.end(), [&](const double& value) {
        double diff = value - mean;
        // Accumulate the squared differences
        variance = variance + (diff * diff); 
    });

    return variance;
}

double calculateCVMultiThreadNonVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;

    // Calculate the rest of mean.
    mean = calcMean(sumMeanMultiThreadNonVectorized(data), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Calculate the rest of mean.
    variance = calcVar(sumVarMultiThreadNonVectorized(data, mean), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double sumMeanParallelVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    const size_t n = data.size();

    __m256d sum_vec = _mm256_setzero_pd(), val, local_sum_vec;

    // Parallel computation of sum
    #pragma omp parallel
    {
        // Local sums for each thread
        local_sum_vec = _mm256_setzero_pd();

        // Perform parallel reduction with AVX
        #pragma omp for
        for (size_t i = 0; i < n; i += 4) {
            // Load 4 double values into an AVX register from the data vector.
            val = _mm256_loadu_pd(&data[i]);

            // Accumulate values into local_sum_vec.
            local_sum_vec = _mm256_add_pd(local_sum_vec, val);
        }

        // Combine the local sums into the global sum (using critical section)
        #pragma omp critical
        {
            sum_vec = _mm256_add_pd(sum_vec, local_sum_vec);
        }
    }

    // Extract the sum from the AVX register and calculate mean
    double temp[4] alignas(32);
    _mm256_store_pd(temp, sum_vec);
    return temp[0] + temp[1] + temp[2] + temp[3];
}

double sumVarParallelVectorized(const std::vector<double>& data, const double mean)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    const size_t n = data.size();

    __m256d var_vec = _mm256_setzero_pd(), 
            val, diff, constant_vec, local_var_vec;

    // Save the mean for parallel computation of variance.
    constant_vec = _mm256_set1_pd(mean);

    // Parallel computation of sum and variance
    #pragma omp parallel
    {
        // Local sums for each thread
        local_var_vec = _mm256_setzero_pd();

        // Perform parallel reduction with AVX
        #pragma omp for
        for (size_t i = 0; i < n; i += 4) {
            // Load 4 double values into an AVX register from the data vector.
            val = _mm256_loadu_pd(&data[i]);

            // Compute the difference from the mean (to be calculated later).
            diff = _mm256_sub_pd(val, constant_vec);
            // Accumulate into local_var_vec the square of diff.
            local_var_vec = _mm256_add_pd(local_var_vec, _mm256_mul_pd(diff, diff));
        }

        // Combine the local sums into the global sum (using critical section)
        #pragma omp critical
        {
            var_vec = _mm256_add_pd(var_vec, local_var_vec);
        }
    }

    // Extract the sum from the AVX register and calculate mean
    double temp[4] alignas(32);
    // Loading of variance to array.
    _mm256_store_pd(temp, var_vec);

    return temp[0] + temp[1] + temp[2] + temp[3];
}

double calculateCVParallelVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    const size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;
    
    mean = calcMean(sumMeanParallelVectorized(data), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Calculate variance
    variance = calcVar(sumVarParallelVectorized(data, mean), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double sumMeanOnGPU(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    const size_t n = data.size();
    const size_t local_size = 256;   // Work-group size (depends on your device)
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size; // Round up

    double mean = 0.0;

    try {
        // Krok 1: Vyberte OpenCL platformu a zařízení
        cl::Platform platform = getPlatform();
        cl::Device device = getDevice(platform);

        // Krok 2: Vytvořte OpenCL kontext a frontu
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Krok 3: Načtěte a sestavte kernel
        std::string kernel_code = loadKernel("../kernels/mykernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.length()));
        cl::Program program(context, sources);
        program.build({device});

        // Step 1: Allocate buffers
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * n, (void*)data.data());
        cl::Buffer buffer_partial_sum(context, CL_MEM_WRITE_ONLY, sizeof(double) * (global_size / local_size));
        cl::Buffer buffer_mean(context, CL_MEM_WRITE_ONLY, sizeof(double));
        cl::Buffer buffer_variance(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

        // Step 2: Set up the kernel
        cl::Kernel kernel(program, "sum_reduction");
        kernel.setArg(0, buffer_data);
        kernel.setArg(1, buffer_partial_sum);
        kernel.setArg(2, static_cast<int>(n));

        // Step 3: Execute the kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
        queue.finish();

        // Step 4: Retrieve the partial sums from the device
        std::vector<double> partial_sums(global_size / local_size);
        queue.enqueueReadBuffer(buffer_partial_sum, CL_TRUE, 0, sizeof(double) * partial_sums.size(), partial_sums.data());

        // Sum the partial sums on the host to get the final result
        for (double partial_sum : partial_sums) {
            mean += partial_sum;
        }

        return mean;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

double sumVarOnGPU(const std::vector<double>& data, const double mean)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    const size_t n = data.size();
    const size_t local_size = 256;   // Work-group size (depends on your device)
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size; // Round up

    double variance = 0.0f;

    try {
        // Krok 1: Vyberte OpenCL platformu a zařízení
        cl::Platform platform = getPlatform();
        cl::Device device = getDevice(platform);

        // Krok 2: Vytvořte OpenCL kontext a frontu
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Krok 3: Načtěte a sestavte kernel
        std::string kernel_code = loadKernel("../kernels/mykernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.length()));
        cl::Program program(context, sources);
        program.build({device});

        // Step 1: Allocate buffers
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * n, (void*)data.data());
        cl::Buffer buffer_partial_sum(context, CL_MEM_WRITE_ONLY, sizeof(double) * (global_size / local_size));
        cl::Buffer buffer_mean(context, CL_MEM_WRITE_ONLY, sizeof(double));
        cl::Buffer buffer_variance(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

        // Step 3: Calculate Variance
        cl::Kernel kernel_var(program, "compute_variance");
        kernel_var.setArg(0, buffer_data);
        kernel_var.setArg(1, mean);
        kernel_var.setArg(2, buffer_variance);
        kernel_var.setArg(3, static_cast<int>(n));

        queue.enqueueNDRangeKernel(kernel_var, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
        queue.finish();

        // Read partial results for variance
        std::vector<double> partial_vars(global_size / local_size);
        queue.enqueueReadBuffer(buffer_variance, CL_TRUE, 0, sizeof(double) * partial_vars.size(), partial_vars.data());

        // Calculate final variance on host
        variance = std::accumulate(partial_vars.begin(), partial_vars.end(), 0.0);

        return variance;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

double calculateCVOnGPU(std::vector<double> data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    const size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;
    
    mean = calcMean(sumMeanOnGPU(data), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Calculate variance
    variance = calcVar(sumVarOnGPU(data, mean), n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double calculateMAD(const std::vector<double>& data,const CalcType calcType)
{
    switch (calcType) {
        case Serial:
            std::cout << "Provedu výpočet MAD sériově." << std::endl;
            return calculateMADSerial(data);
            break;
        case Vectorized:
            std::cout << "Provedu výpočet MAD vektorizovaně." << std::endl;
            return calculateMADVectorized(data);
            break;
        case MultiThreadNonVectorized:
            std::cout << "Provedu výpočet MAD vícevláknově, nevektorizovaně." << std::endl;
            return calculateMADMultiThreadNonVectorized(data);
            break;
        case ParallelVectorized:
            std::cout << "Provedu výpočet MAD vícevláknově, vektorizovaně." << std::endl;
            return calculateMADParallelVectorized(data);
            break;
        default:
            std::cout << "Provedu výpočet MAD na GPU." << std::endl;
            return calculateMADOnGPU(data);
            break;
    }
}

double calculate_median(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    return getMedian(data);
}

std::vector<double> sortDataSerial(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    std::vector<double> sorted(data);
    std::sort(sorted.begin(), sorted.end());

    return sorted;
}

std::vector<double> calculateAbsDevSerial(const std::vector<double>& data, const double median)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    std::vector<double> deviations = {};

    // For each value of input data lets calculate the deviation.
    for (double value : data) {
        deviations.push_back(std::fabs(value - median));
    }

    return sortDataSerial(deviations);
}

double calculateMADSerial(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> sorted = sortDataSerial(data);

    // Get the median from sorted input data.
    double median = getMedian(sorted);

    std::vector<double> deviations = calculateAbsDevSerial(data, median);

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

std::vector<double> sortDataVectorized(const std::vector<double>& data) 
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    std::vector<double> sortedData(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(sortedData.begin(), sortedData.end());

    return sortedData;
}

std::vector<double> calculateAbsDevVectorized(const std::vector<double>& data, const double median) 
{ 
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    size_t i = 0;

    __m256d sign_mask, vec_data, vec_median, vec_dev;

    std::vector<double> deviations(data.size());

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
        deviations[i] = std::fabs(data[i] - median);
    }

    return sortDataVectorized(deviations);
}

double calculateMADVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> sortedData = sortDataVectorized(data);

    // Get the median from sorted input data.
    double median = getMedian(sortedData);

    std::vector<double> deviations = calculateAbsDevVectorized(data, median);
    if (deviations.size() == 0) {
        throw std::invalid_argument(EMPTY_DEVIATIONS_MESSAGE);
    }

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

std::vector<double> sortDataMultiThreadNonVectorized(const std::vector<double>& data) 
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    std::vector<double> sortedData(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(sortedData.begin(), sortedData.end());

    return sortedData;
}

std::vector<double> calculateAbsDevMultiThreadNonVectorized(const std::vector<double>& data, const double median) 
{ 
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    const size_t n = data.size();

    std::vector<double> deviations(n);

    // For each value of input data lets calculate the deviation.
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        deviations[i] = std::fabs(data[i] - median);
    }

    return sortDataMultiThreadNonVectorized(deviations);
}

double calculateMADMultiThreadNonVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> sortedData = sortDataMultiThreadNonVectorized(data);

    // Get the median from sorted input data.
    const double median = getMedian(sortedData);

    std::vector<double> deviations = calculateAbsDevMultiThreadNonVectorized(data, median);
    if (deviations.size() == 0) {
        throw std::invalid_argument(EMPTY_DEVIATIONS_MESSAGE);
    }

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

std::vector<double> sortDataParallelVectorized(const std::vector<double>& data) {
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }
    
    std::vector<double> sortedData(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(sortedData.begin(), sortedData.end());

    return sortedData;
}

std::vector<double> calculateAbsDevParallelVectorized(const std::vector<double>& data, const double median) {
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }
    
    std::vector<double> deviations(data.size());
    
    __m256d sign_mask = _mm256_set1_pd(-0.0), vec_data, vec_median, vec_dev; // -0.0 in IEEE 754 has the sign bit set

    #pragma omp parallel for
    for (size_t i = 0; i + 4 <= data.size(); i += 4) {
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
    #pragma omp parallel for
    for (size_t i = (data.size() / 4) * 4; i < data.size(); ++i) {
        deviations[i] = std::fabs(data[i] - median);
    }

    return sortDataParallelVectorized(deviations);
}

double calculateMADParallelVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> sortedData = sortDataParallelVectorized(data);

    // Get the median from sorted input data.
    double median = getMedian(sortedData);

    std::vector<double> deviations = calculateAbsDevParallelVectorized(data, median);
    if (deviations.size() == 0) {
        throw std::invalid_argument(EMPTY_DEVIATIONS_MESSAGE);
    }

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

std::vector<double> sortDataOnGPU(const std::vector<double>& data) 
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    std::vector<double> sortedData(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(sortedData.begin(), sortedData.end());

    return sortedData;
}

std::vector<double> calculateAbsDevOnGPU(const std::vector<double>& data, const double median) 
{   
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    const size_t n = data.size();

    std::vector<double> deviations(n);

    try {
        cl::Platform platform = getPlatform();
        cl::Device device = getDevice(platform);
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Načtení kernelu
        std::string kernel_code = loadKernel("../kernels/mykernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.size()));
        cl::Program program(context, sources);
        program.build({device});

        // Předání dat do GPU
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), (void*)data.data());
        cl::Buffer buffer_deviations(context, CL_MEM_WRITE_ONLY, n * sizeof(double));

        // Inicializace kernelu a nastavení argumentů
        cl::Kernel kernel(program, "calculate_absolute_deviation");
        kernel.setArg(0, buffer_data);
        kernel.setArg(1, buffer_deviations);
        kernel.setArg(2, median);

        // Spuštění kernelu
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
        queue.finish();

        // Načtení výsledků zpět na hostitele
        queue.enqueueReadBuffer(buffer_deviations, CL_TRUE, 0, n * sizeof(double), deviations.data());

        return sortDataOnGPU(deviations);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return {};
    }
}

double calculateMADOnGPU(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Sort input data so we can find the median as the value in the middle.
    std::vector<double> sortedData = sortDataOnGPU(data);

    // Get the median from sorted input data.
    double median = getMedian(sortedData);

    std::vector<double> deviations = calculateAbsDevOnGPU(data, median);
    if (deviations.size() == 0) {
        throw std::invalid_argument(EMPTY_DEVIATIONS_MESSAGE);
    }

    // Get the median from sorted deviations.
    return getMedian(deviations);
}
