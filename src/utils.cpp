#include "utils.h"
#include "cl.hpp"
#include <iostream>

cl::Platform getPlatform() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    return platforms.front();  // Select the first platform
}

cl::Device getDevice(cl::Platform platform) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);  // Select GPU devices
    return devices.front();  // Select the first device
}
