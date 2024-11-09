cmake_minimum_required(VERSION 3.10)
project(PPR_Project)

set(CMAKE_CXX_STANDARD 17)  # Set C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Include OpenCL headers
include_directories(${OpenCL_INCLUDE_DIRS} include/)

# Source files
set(SOURCES
    src/main.cpp
    src/run_initializer.cpp
    src/kernels.cpp
    src/utils.cpp
    src/data_loader.cpp
    src/calculations.cpp
    src/graph_printer.cpp
    src/sort.cpp
    src/basic_math.cpp
    src/calc_type.cpp
    src/sortings.cpp
    src/performance_stats.cpp
)

set_source_files_properties(
    ${CMAKE_INCLUDE_DIR}/cl.h
    PROPERTIES COMPILE_FLAGS "-w"
)

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
