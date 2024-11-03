#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>
#include "kernels.h"

cl::Platform getPlatform();

cl::Device getDevice(cl::Platform platform);


cl::Program::Sources getSources();