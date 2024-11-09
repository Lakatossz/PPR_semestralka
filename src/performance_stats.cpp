#include "../include/performance_stats.h"

void PerformanceStats::startTimer() {
    startTime = std::chrono::high_resolution_clock::now();
}

void PerformanceStats::stopTimer() {
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime - startTime).count();
    durations.push_back(duration);
}

void PerformanceStats::printHeadSummary() const {
    // Open file in read/write mode with truncation (clear content)
    std::fstream file(STATS_FILE_NAME, std::ios::in | std::ios::out | std::ios::trunc);
    
    if (file.is_open()) {
        // File content is now cleared
        std::cout << "File cleared successfully." << std::endl;
        file << STATS_HEADER;
    } else {
        std::cerr << "Failed to open the file." << std::endl;
    }
    // File automatically closed when the fstream goes out of scope
}

// Print summary
void PerformanceStats::printSummary(const std::string& calcType) const {

    std::ofstream file(STATS_FILE_NAME, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << STATS_FILE_NAME << std::endl;
    }

    file << calcType << "; ";
    auto endTime = std::chrono::high_resolution_clock::now();
    file << std::chrono::duration<double>(endTime - startTime).count() << std::endl;
}