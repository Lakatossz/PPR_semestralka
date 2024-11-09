#pragma once

#include <string>

enum CalcType {
    Serial,
    Vectorized,
    MultiThreadNonVectorized,
    ParallelVectorized,
    OnGPU
};

std::string calcTypeToString(CalcType calcType);