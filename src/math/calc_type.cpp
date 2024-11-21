#include "../../include/math/calc_type.h"

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
