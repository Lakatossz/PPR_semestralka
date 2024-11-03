#include <iostream>
#include "utils.h"
#include "data_loader.h"
#include "calculations.h"
#include "graph_printer.h"

#define NUMBER_OF_VALUES 10000//74380
#define NUMBER_OF_PATIENTS 16
#define NUMBER_OF_DIMENSIONS 3

const std::vector<std::string> files = {"ACC_001.csv", "ACC_002.csv", "ACC_003.csv", "ACC_004.csv", "ACC_005.csv", "ACC_006.csv", "ACC_007.csv", "ACC_008.csv", "ACC_009.csv", "ACC_010.csv", "ACC_011.csv", "ACC_012.csv", "ACC_013.csv", "ACC_014.csv", "ACC_015.csv", "ACC_016.csv"};

const std::vector<std::string> dimensions = {"x", "y", "z"};

int main() {
    CalcType calcType = CalcType::OnGPU;

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
    
    for (size_t i = 0; i < 1; ++i) {
        size_t lines = countLines("../data/" + files[i]);
        file << "Oteviram soubor " << "../data/" + files[i] << "s " << lines << " řádky." << std::endl;
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
