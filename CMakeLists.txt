cmake_minimum_required(VERSION 3.10)
project(PPR_Project)

set(CMAKE_CXX_STANDARD 17)  # Set C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Include directories for headers
include_directories(${OpenCL_INCLUDE_DIRS} include/)

# Source files
set(SOURCES
    src/main.cpp
    src/kernels.cpp
    src/utils.cpp
    src/data_loader.cpp
    src/calculations.cpp
    src/graph_printer.cpp
    src/sort.cpp
)

link_libraries(${OpenCL_LIBRARIES})

# Add the executable
add_executable(PPR_Project ${SOURCES})

target_compile_options(PPR_Project PRIVATE -mavx2 -O2)

# Link the OpenCL library
target_link_libraries(PPR_Project ${OpenCL_LIBRARIES})
