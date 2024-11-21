#include "utils.h"
#define EMPTY_VECTOR_MESSAGE "Data vector is empty"

cl::Platform getPlatform() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    return platforms.front();  // Select the first platform
}

cl::Device getDevice(cl::Platform platform) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);  // Select GPU devices
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found.");
    }

    return devices.front();  // Select the first device
}

cl::Context getContext() {
    // Krok 1: Vyberte OpenCL platformu a zařízení
    cl::Platform platform = getPlatform();
    cl::Device device = getDevice(platform);
    // Krok 2: Vytvořte OpenCL kontext a frontu
    return cl::Context(device);
}

cl::Context prepareForGPU(cl::CommandQueue &queue, cl::Program &program) {
    // Krok 1: Vyberte OpenCL platformu a zařízení
        cl::Platform platform = getPlatform();
        cl::Device device = getDevice(platform);

        // Krok 2: Vytvořte OpenCL kontext a frontu
        cl::Context context(device);
        queue = cl::CommandQueue(context, device);

        // Krok 3: Načtěte a sestavte kernel
        std::string kernel_code = loadKernel("../kernels/mykernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.length()));
        program = cl::Program(context, sources);
        program.build({device});

        return context;
}

cl::Buffer prepareDataBuffer(const cl::Context context, const std::vector<double>& data) {
    return cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * data.size(), (void*)data.data());
}

void sumMeanUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_partial_sum, size_t global_size, size_t local_size, int n)
 {
    cl::Kernel kernel(program, "sum_reduction");
    kernel.setArg(0, buffer_data);
    kernel.setArg(1, buffer_partial_sum);
    kernel.setArg(2, static_cast<int>(n));
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    queue.finish();
}

void sumVarUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_variance, double mean, size_t global_size, size_t local_size, int n)
 {
    cl::Kernel kernel(program, "compute_variance");
    kernel.setArg(0, buffer_data);
    kernel.setArg(1, mean);
    kernel.setArg(2, buffer_variance);
    kernel.setArg(3, static_cast<int>(n));
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    queue.finish();
 }

void sortDataUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_sorted, int n)
 {
    cl::Kernel kernel(program, "calculate_absolute_deviation");
    kernel.setArg(0, buffer_data);
    kernel.setArg(1, buffer_sorted);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n));
    queue.finish();
}

// Example function that performs operations on buffer_data
void calculateAbsoluteDeviationUsingGPUKernel(cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_data, cl::Buffer &buffer_deviations, double median, int n) {
    cl::Kernel kernel(program, "calculate_absolute_deviation");
    kernel.setArg(0, buffer_data);
    kernel.setArg(1, buffer_deviations);
    kernel.setArg(2, median);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n));
    queue.finish();
}
