#include "../../include/sort/sortings.h"

std::vector<double> sortData(const std::vector<double>& data, const CalcType calcType, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat)
{
    stat.stopTimer();

    switch (calcType) {
        case Serial: {
            std::vector<double> sorted = std::vector(data);

            size_t originalSize = sorted.size();

            bitonicSort(sorted, true);

            return sorted;
            //return sortDataWithPolicy(data, std::execution::seq, stat);
            break;
        }
        case Vectorized:
            return sortDataVectorized(data, stat);
            break;
        case MultiThreadNonVectorized:
            return sortDataWithPolicy(data, std::execution::par, stat);
            break;
        case ParallelVectorized:
            return sortDataWithPolicy(data, std::execution::par_unseq, stat);
            break;
        default:
            return sortDataOnGPU(data, queue, program, buffer_data, stat);
            break;
    }
}

template <typename ExecutionPolicy>
std::vector<double> sortDataWithPolicy(const std::vector<double>& data, const ExecutionPolicy policy, PerformanceStats& stat)
{
    std::vector<double> sortedData(data);
    std::sort(policy, sortedData.begin(), sortedData.end());

    return sortedData;
}

std::vector<double> sortDataVectorized(const std::vector<double>& data, PerformanceStats& stat) 
{
    std::vector<double> sortedData(data);

    // Sort input data so we can find the median as the value in the middle.
    std::sort(std::execution::unseq, sortedData.begin(), sortedData.end());

    return sortedData;
}

std::vector<double> sortDataOnGPU(const std::vector<double>& data, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, PerformanceStats& stat) 
{
    const size_t n = data.size();

    std::vector<double> sortedData(data);

    try {
        cl::Context context = getContext();

        /*cl::Buffer sorted_buffer(context, CL_MEM_WRITE_ONLY, n * sizeof(double));
        sortDataUsingGPUKernel(queue, program, buffer_data, sorted_buffer, n);
        queue.enqueueReadBuffer(sorted_buffer, CL_TRUE, 0, n * sizeof(double), sortedData.data());*/

        std::sort(sortedData.begin(), sortedData.end());

        return sortedData;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error (calculateAbsDevOnGPU): " << err.what() << "(" << err.err() << ")" << std::endl;
        return {};
    }
}
