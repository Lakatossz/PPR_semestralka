#include "../include/run_initializer.h"

int initialize_run() {

    auto startTime = std::chrono::high_resolution_clock::now();

    CalcType calcType = CalcType::MultiThreadNonVectorized;
    std::array<CalcType, NUMBER_OF_CALCULATION_TYPES> calcTypeArray = {
        CalcType::Serial, 
        CalcType::Vectorized, 
        CalcType::MultiThreadNonVectorized, 
        CalcType::ParallelVectorized, 
        CalcType::OnGPU
    };

    for (auto& calcType : calcTypeArray) {
        std::string filename = OUTPUT_FILE_NAME_HEAD + calcTypeToString(calcType) + OUTPUT_FILE_NAME_TAIL;
        std::fstream file(filename, std::ios::in | std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
        }

        file << "CV (%);MAD" << std::endl;

        file.close();
    }

    std::array<std::ofstream, NUMBER_OF_CALCULATION_TYPES> output_files;

    cl::CommandQueue queue = {};
    cl::Program program = {};

    cl::Context context = prepareForGPU(queue, program);

    std::array<PerformanceStats, NUMBER_OF_CALCULATION_TYPES> stats;
    //stats.printHeadSummary();

    vector<vector<double>> data = {};
    double cv[NUMBER_OF_PATIENTS][NUMBER_OF_DIMENSIONS] = {};
    double mad[NUMBER_OF_PATIENTS][NUMBER_OF_DIMENSIONS] = {};

    for (size_t file_index = 0; file_index < files.size() / 2; ++file_index) {
        size_t lines = DEBUG_MODE == 1 ? DEBUG_NUMBER_OF_VALUES : countLines(DATA_FOLDER_NAME + files[file_index]);
        std::cout << "Oteviram soubor " << DATA_FOLDER_NAME + files[file_index] << " s " << lines << " řádky." << std::endl;
        data = read(DATA_FOLDER_NAME + files[file_index], lines);

        for (size_t i = 0; i < calcTypeArray.size(); ++i) {
            auto& calcType = calcTypeArray[i];
            auto& stat = stats[i];

            std::string filename = OUTPUT_FILE_NAME_HEAD + calcTypeToString(calcType) + OUTPUT_FILE_NAME_TAIL;
            std::ofstream file(filename, std::ios::app);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
            }

            stat.startTimer();

            if (data.size() > 0) {
                for (size_t dimension_index = 0; dimension_index < NUMBER_OF_DIMENSIONS; ++dimension_index) {

                    // Step 1: Allocate buffers
                    cl::Buffer buffer_data = prepareDataBuffer(context, data[dimension_index]);

                    stat.stopTimer();

                    cv[file_index][dimension_index] = calculateCV(data[dimension_index], calcType, queue, program, buffer_data, stat);
                    
                    stat.stopTimer();

                    mad[file_index][dimension_index] = calculateMAD(data[dimension_index], calcType, queue, program, buffer_data, stat);
                
                    stat.stopTimer();

                    file <<  cv[file_index][dimension_index] << ";";
                    file << mad[file_index][dimension_index] << std::endl;

                    stat.stopTimer();

                    //cv[file_index][dimension_index] = calculateCV(data[dimension_index], CalcType::ParallelVectorized, queue, program, buffer_data, stat);
                    //mad[file_index][dimension_index] = calculateMAD(data[dimension_index], CalcType::ParallelVectorized, queue, program, buffer_data, stat);
                
                    //file <<  cv[file_index][dimension_index] << ";";
                    //file << mad[file_index][dimension_index] << std::endl;

                    //stat.stopTimer();
                }
                
                // Print the summary
                stat.printSummary(calcTypeToString(calcType));
                stat.clearTimer();
            }

            // Print the summary
            //stat.printSummary(calcTypeToString(CalcType::ParallelVectorized));
        
            file.close();
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << "Program běžel (s): " << duration << std::endl;
    
    return 0;
}