#include "utils.h"
#include "cl.hpp"
#include <iostream>

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

cl::Program::Sources getSources() {
    std::string kernel_code = loadKernel("../kernels/mykernel.cl");
    return cl::Program::Sources(1, std::make_pair(kernel_code.c_str(), kernel_code.size()));
}
