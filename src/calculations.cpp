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

double calculateCVSerial(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    const size_t n = data.size();

    double mean = 0.0, variance = 0.0, stddev = 0.0;

    // First we need to get the mean of the input data.
    mean = calcMean(std::accumulate(data.begin(), data.end(), 0.0), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Than we need to get the variance from calculated mean and each value.
    for (const auto& value : data) {
        double diff = value - mean;

        // Accumulate the squared differences
        variance += (diff) * (diff);
    }
    // Finalize variance calculation
    variance = calcVar(variance, n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
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
    mean = calcMean((temp[0] + temp[1] + temp[2] + temp[3]), n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

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
    _mm256_store_pd(temp, var_vec);

    // Calculate the rest of mean.
    variance = calcVar(temp[0] + temp[1] + temp[2] + temp[3], n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double calculateCVMultiThreadNonVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    double mean = 0.0, stddev = 0.0;
    std::atomic<double> variance(0.0);

    // First we need to get the mean of the input data.
    mean = calcMean(std::reduce(std::execution::par, data.begin(), data.end(), 0.0), data.size());
    
    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Than we need to get the variance from calculated mean and each value.
    std::for_each(std::execution::par, data.begin(), data.end(), [&](const double& value) {
        double diff = value - mean;
        // Accumulate the squared differences
        variance = variance + (diff * diff); 
    });
    // Finalize variance calculation
    variance = calcVar(variance, data.size());

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double calculateCVVectorized_(const std::vector<double>& data) {
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Size of input data.
    size_t n = data.size();
    
    double mean = 0.0, variance = 0.0, stddev = 0.0;
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d var_vec = _mm256_setzero_pd();
    
    // Parallel computation of sum and variance
    #pragma omp parallel
    {
        // Local sums for each thread
        __m256d local_sum_vec = _mm256_setzero_pd();
        __m256d local_var_vec = _mm256_setzero_pd();

        // Perform parallel reduction with AVX
        #pragma omp for
        for (size_t i = 0; i < n; i += 4) {
            // Load 4 double values into an AVX register from the data vector.
            __m256d val = _mm256_loadu_pd(&data[i]);

            // Accumulate values into local_sum_vec.
            local_sum_vec = _mm256_add_pd(local_sum_vec, val);

            // Compute the difference from the mean (to be calculated later).
            __m256d diff = _mm256_sub_pd(val, _mm256_set1_pd(0.0)); // Placeholder for mean
            // Accumulate into local_var_vec the square of diff.
            local_var_vec = _mm256_add_pd(local_var_vec, _mm256_mul_pd(diff, diff));
        }

        // Combine the local sums into the global sum (using critical section)
        #pragma omp critical
        {
            sum_vec = _mm256_add_pd(sum_vec, local_sum_vec);
            var_vec = _mm256_add_pd(var_vec, local_var_vec);
        }
    }

    // Extract the sum from the AVX register and calculate mean
    double temp[4] alignas(32); // Ensure 32-byte alignment
    _mm256_store_pd(temp, sum_vec);
    mean = calcMean(temp[0] + temp[1] + temp[2] + temp[3], n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

    // Now compute variance using the mean
    // Reset local variance sum since we already accumulated it
    double var_sum = 0.0;
    _mm256_store_pd(temp, var_vec);
    var_sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Calculate variance
    variance = calcVar(var_sum, n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
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
    __m256d sum_vec = _mm256_setzero_pd(), var_vec = _mm256_setzero_pd(), 
            val, diff, constant_vec, local_sum_vec, local_var_vec;

    // Parallel computation of sum
    #pragma omp parallel
    {
        // Local sums for each thread
        local_sum_vec = _mm256_setzero_pd();
        local_var_vec = _mm256_setzero_pd();

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
    mean = calcMean(temp[0] + temp[1] + temp[2] + temp[3], n);

    // If mean is close to zero, return 0 (CV would be very large or undefined)
    if (std::abs(mean) < ALMOST_ZERO) {
        return 0.0;
    }

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

    // Loading of variance to array.
    _mm256_store_pd(temp, var_vec);

    // Calculate variance
    variance = calcVar(temp[0] + temp[1] + temp[2] + temp[3], n);

    // Now get the Standard deviation out of variance.
    stddev = std::sqrt(variance);

    // Return final Coefficient of Variation value in percentage.
    return calcCV(mean, stddev);
}

double calculateCVOnGPU(std::vector<double> data)
{
    const size_t n = data.size();

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

        // Krok 4: Vytvořte paměťové buffery
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, data.data());
        cl::Buffer buffer_mean(context, CL_MEM_WRITE_ONLY, sizeof(float));
        cl::Buffer buffer_variance(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        // Krok 5: Nastavení kernelu a výpočet průměru
        cl::Kernel kernel_mean(program, "compute_mean");
        kernel_mean.setArg(0, buffer_data);
        kernel_mean.setArg(1, buffer_mean);
        kernel_mean.setArg(2, static_cast<int>(n));

        queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(n), cl::NullRange);
        queue.finish();

        // Získání výsledku průměru
        float mean;
        queue.enqueueReadBuffer(buffer_mean, CL_TRUE, 0, sizeof(float), &mean);

        // Krok 6: Nastavení kernelu a výpočet rozptylu
        cl::Kernel kernel_variance(program, "compute_variance");
        kernel_variance.setArg(0, buffer_data);
        kernel_variance.setArg(1, buffer_mean);
        kernel_variance.setArg(2, buffer_variance);
        kernel_variance.setArg(3, static_cast<int>(n));

        queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(n), cl::NullRange);
        queue.finish();

        // Získání výsledku rozptylu
        std::vector<float> variance(n);
        queue.enqueueReadBuffer(buffer_variance, CL_TRUE, 0, sizeof(float) * n, variance.data());

        float total_variance = 0.0f;
        for (float v : variance) {
            total_variance += v;
        }

        // Směrodatná odchylka
        float stddev = std::sqrt(total_variance);

        // Výpočet koeficientu variance
        float cv = (stddev / mean) * 100.0f;

        std::cout << "Mean: " << mean << std::endl;
        std::cout << "Standard Deviation: " << stddev << std::endl;
        std::cout << "Coefficient of Variation (CV): " << cv << "%" << std::endl;

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
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

double calculateMADSerial(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> deviations(data.size()), temp(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(temp.begin(), temp.end());

    // Get the median from sorted input data.
    double median = getMedian(temp);

    // For each value of input data lets calculate the deviation.
    for (double value : temp) {
        deviations.push_back(std::fabs(value - median));
    }

    // Sort deviations so we can find the median as the value in the middle.
    std::sort(deviations.begin(), deviations.end());

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

double calculateMADVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> deviations(data.size()), temp(data);

    double median = 0.0;

    size_t i = 0;

    __m256d sign_mask, vec_data, vec_median, vec_dev;

    /* TODO needs to be vectorized. */
    // Sort input data so we can find the median as the value in the middle.
    std::sort(temp.begin(), temp.end());

    // Get the median from sorted input data.
    median = getMedian(temp);

    // Mask for setting abs value in vector (IEEE 754 has the sign bit set).
    sign_mask = _mm256_set1_pd(-0.0);

    // Now we process in chunks of 4 doubles (4 64-bit doubles in a 256-bit register).
    for (; i + 4 <= temp.size(); i += 4) {
        // Load 4 double values into an AVX register from the data vector.
        vec_data = _mm256_loadu_pd(&temp[i]);

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
    for (; i < temp.size(); ++i) {
        deviations[i] = std::fabs(temp[i] - median);
    }

    /* TODO needs to be vectorized. */
    // Sort deviations so we can find the median as the value in the middle.
    std::sort(deviations.begin(), deviations.end());

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

double calculateMADMultiThreadNonVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> deviations(data.size()), temp(data);

    /* TODO needs to be parallelized. */
    // Sort input data so we can find the median as the value in the middle.
    std::sort(temp.begin(), temp.end());

    // Get the median from sorted input data.
    const double median = getMedian(temp);

    // For each value of input data lets calculate the deviation.
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        deviations[i] = std::fabs(data[i] - median);
    }

    /* TODO needs to be parallelized. */
    // Sort deviations so we can find the median as the value in the middle.
    std::sort(deviations.begin(), deviations.end());

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

double calculateMADParallelVectorized(const std::vector<double>& data)
{
    // Edge case: if data is empty, return 0 or throw an exception
    if (data.empty()) {
        throw std::invalid_argument(EMPTY_VECTOR_MESSAGE);
    }

    // Initialization of vector for deviations and temp for sorting.
    std::vector<double> deviations(data.size()), temp(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(temp.begin(), temp.end());

    // Get the median from sorted input data.
    double median = getMedian(temp);

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

    /* TODO needs to be vectorized. */
    // Sort deviations so we can find the median as the value in the middle.
    std::sort(deviations.begin(), deviations.end());

    // Get the median from sorted deviations.
    return getMedian(deviations);
}

double calculateMADOnGPU(const std::vector<double>& data)
{
    const size_t n = data.size();

    try {
        // Inicializace OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found.");
        }

        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Načtení kernelu
        std::string kernel_code = loadKernel("mykernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.size()));
        cl::Program program(context, sources);
        program.build({device});

        // Výpočet mediánu na hostitelském zařízení
        double median = calculate_median(const_cast<std::vector<double>&>(data));

        // Předání dat do GPU
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.size() * sizeof(double), (void*)data.data());
        cl::Buffer buffer_deviations(context, CL_MEM_WRITE_ONLY, data.size() * sizeof(double));

        // Inicializace kernelu a nastavení argumentů
        cl::Kernel kernel(program, "calculate_absolute_deviation");
        kernel.setArg(0, buffer_data);
        kernel.setArg(1, buffer_deviations);
        kernel.setArg(2, median);

        // Spuštění kernelu
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(data.size()), cl::NullRange);
        queue.finish();

        // Načtení výsledků zpět na hostitele
        std::vector<double> deviations(data.size());
        queue.enqueueReadBuffer(buffer_deviations, CL_TRUE, 0, data.size() * sizeof(double), deviations.data());

        // Výpočet mediánu z absolutních odchylek
        return calculate_median(deviations);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return -1.0;
    }
}
