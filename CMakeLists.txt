cmake_minimum_required(VERSION 3.14)
project(PPR_Project)

set(CMAKE_CXX_STANDARD 17)  # Set C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Find TBB package
find_package(TBB REQUIRED)

# Include OpenCL headers
include_directories(${OpenCL_INCLUDE_DIRS} include/gpu/)

# Include TBB headers
include_directories(${TBB_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/run_initializer.cpp
    src/gpu/kernels.cpp
    src/gpu/utils.cpp
    src/sort/sortings.cpp
    src/utils/data_loader.cpp
    src/math/calculations.cpp
    src/utils/graph_printer.cpp
    src/sort/bitonic_sort.cpp
    src/math/basic_math.cpp
    src/math/calc_type.cpp
    src/utils/performance_stats.cpp
)

set_source_files_properties(
    ${CMAKE_INCLUDE_DIR}/gpu/cl.h
    PROPERTIES COMPILE_FLAGS "-w"
)

add_compile_definitions(CL_TARGET_OPENCL_VERSION=120) # Adjust the version as needed

# Add the executable
add_executable(PPR_Project ${SOURCES})

# Link OpenMP library to your project target if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(PPR_Project PUBLIC OpenMP::OpenMP_CXX)
endif()

# Specify compiler flags for target
target_compile_options(PPR_Project PRIVATE -mavx2 -O3 -fopenmp -march=native)

# Link OpenCL library
target_link_libraries(PPR_Project PRIVATE ${OpenCL_LIBRARIES})

# Link TBB library
target_link_libraries(PPR_Project PRIVATE TBB::tbb)
