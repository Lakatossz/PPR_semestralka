#pragma once

#include <iostream>
#include "gpu/utils.h"
#include "utils/data_loader.h"
#include "math/calculations.h"
#include "utils/graph_printer.h"
#include "utils/performance_stats.h"

#define NUMBER_OF_VALUES 10000
#define NUMBER_OF_PATIENTS 16
#define NUMBER_OF_DIMENSIONS 3

#define NUMBER_OF_CALCULATION_TYPES 5

#define DATA_FOLDER_NAME "../data/"
#define OUTPUT_FILE_NAME_HEAD "../output/vysledek_"
#define OUTPUT_FILE_NAME_TAIL ".csv"

#define DEBUG_MODE 1
#define DEBUG_NUMBER_OF_VALUES 16
#define DEBUG_NUMBER_OF_PATIENTS 1
#define DEBUG_NUMBER_OF_DIMENSIONS 3

const std::array<std::string, NUMBER_OF_PATIENTS> files = {"ACC_001.csv", "ACC_002.csv", "ACC_003.csv", "ACC_004.csv", "ACC_005.csv", "ACC_006.csv", "ACC_007.csv", "ACC_008.csv", "ACC_009.csv", "ACC_010.csv", "ACC_011.csv", "ACC_012.csv", "ACC_013.csv", "ACC_014.csv", "ACC_015.csv", "ACC_016.csv"};

const std::array<std::string, NUMBER_OF_DIMENSIONS> dimensions = {"x", "y", "z"};

int initialize_run();