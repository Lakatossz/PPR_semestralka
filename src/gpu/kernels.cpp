#include "kernels.h"
#include <fstream>
#include <iostream>

std::string loadKernel(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to load kernel from " << path << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
