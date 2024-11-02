#pragma once

#include <cl.hpp>

cl::Platform getPlatform();

cl::Device getDevice(cl::Platform platform);