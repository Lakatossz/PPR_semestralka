#include <iostream>
#include "utils.h"
#include "data_loader.h"
#include "calculations.h"
#include "graph_printer.h"

#define NUMBER_OF_VALUES 150000//74380
#define NUMBER_OF_PATIENTS 16
#define NUMBER_OF_DIMENSIONS 3

const std::vector<std::string> files = {"ACC_001.csv", "ACC_002.csv", "ACC_003.csv", "ACC_004.csv", "ACC_005.csv", "ACC_006.csv", "ACC_007.csv", "ACC_008.csv", "ACC_009.csv", "ACC_010.csv", "ACC_011.csv", "ACC_012.csv", "ACC_013.csv", "ACC_014.csv", "ACC_015.csv", "ACC_016.csv"};

const std::vector<std::string> dimensions = {"x", "y", "z"};

int main() {

    const int N = 1024;
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);

    CalcType calcType = CalcType::ParallelVectorized;

    // Open a file in write mode
    std::ofstream file("../output/vysledek_" + calcTypeToString(calcType) + ".csv");
    std::ofstream file_benchmark("../output/vysledek_" + calcTypeToString(CalcType::Serial) + ".csv");

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write to file using the << operator, similar to std::cout
    file << "Výsledek běhu programu!" << std::endl;
    file << "Právě běžel: " << calcTypeToString(calcType) << std::endl;

    vector<vector<double>> data[NUMBER_OF_PATIENTS] = {};
    double cv[NUMBER_OF_PATIENTS][NUMBER_OF_DIMENSIONS] = {};
    double mad[NUMBER_OF_PATIENTS][NUMBER_OF_DIMENSIONS] = {};
    
    for (size_t i = 0; i < NUMBER_OF_PATIENTS; ++i) {
        file << "Oteviram soubor " << "../data/" + files[i] << std::endl;
        data[i] = read("../data/" + files[i], NUMBER_OF_VALUES);

        if (data[i].size() > 0) {
            file << "CV (%);MAD" << std::endl;
            for (size_t j = 0; j < NUMBER_OF_DIMENSIONS; ++j) {
                cv[i][j] = calculateCV(data[i][j], calcType);
                mad[i][j] = calculateMAD(data[i][j], calcType);
            
                file <<  cv[i][j] << ";";
                file << mad[i][j] << std::endl;

                cv[i][j] = calculateCV(data[i][j], CalcType::Serial);
                mad[i][j] = calculateMAD(data[i][j], CalcType::Serial);
            
                file <<  cv[i][j] << ";";
                file << mad[i][j] << std::endl;
            }
        }
    }
    
    std::cout << "OpenCL computation complete!" << std::endl;
    
    return 0;
}


// try {
        // Initialize OpenCL
        // cl::Platform platform = getPlatform();
        // cl::Device device = getDevice(platform);
        // cl::Context context(device);
        // cl::CommandQueue queue(context, device);

        // // Load kernel
        // std::string kernelSource = loadKernel("kernels/mykernel.cl");
        // cl::Program program(context, kernelSource);
        // program.build({device});

        // Get all platforms and devices
    //     std::vector<cl::Platform> platforms;
    //     cl::Platform::get(&platforms);
    //     cl::Platform platform = platforms.front();

    //     std::vector<cl::Device> devices;
    //     platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    //     cl::Device device = devices.front();

    //     // Create context and command queue
    //     cl::Context context(device);
    //     cl::CommandQueue queue(context, device);

    //     // Create buffers
    //     cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, A.data());
    //     cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, B.data());
    //     cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

    //     // Build program
    //     std::string kernelSource = loadKernel("../kernels/mykernel.cl");
    //     cl::Program program(context, kernelSource);
    //     program.build({device});

    //     // Set kernel arguments
    //     cl::Kernel kernel(program, "vector_add");
    //     kernel.setArg(0, bufferA);
    //     kernel.setArg(1, bufferB);
    //     kernel.setArg(2, bufferC);

    //     // Execute the kernel
    //     queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
    //     queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * N, C.data());

    //     // Output results
    //     std::cout << "First 10 results of vector addition:\n";
    //     for (int i = 0; i < 10; ++i) {
    //         std::cout << "C[" << i << "] = " << C[i] << std::endl;
    //     }

    // } catch (const int &e) {
    //     std::cerr << "OpenCL error: " << std::endl;
    // }

    // Setup data (e.g., buffers, kernel arguments) and execute kernel
    // ...