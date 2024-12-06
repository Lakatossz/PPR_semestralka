#include "../../include/utils/performance_stats.h"

void PerformanceStats::startTimer() {
    startTime = std::chrono::high_resolution_clock::now();
}

void PerformanceStats::stopTimer(const std::string checkPointName) {
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime - startTime).count();
    checkPoint c = {};
    c.checkPointDuration = duration;
    c.checkPointName = checkPointName;
    durations.push_back(c);
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
void PerformanceStats::printSummary(const std::string& calcType, const bool sLabelem) const {

    std::ofstream file(STATS_FILE_NAME, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << STATS_FILE_NAME << std::endl;
    }

    file << "\n" << calcType << "; ";

    for (size_t i = 0; i < durations.size(); ++i) {
        if (sLabelem) {
            file << durations[i].checkPointDuration << " (" << durations[i].checkPointName << "); ";
        } else {
            file << durations[i].checkPointDuration << "; ";
        }
    }
}

void PerformanceStats::clearTimer() {
    durations.clear();
}

void PerformanceStats::clearFile() {
    std::ofstream file(STATS_FILE_NAME, std::ios::out | std::ios::trunc); // Open in truncate mode
    if (file) {
        file.close(); // Close the file after truncating
    } else {
        std::cerr << "Error: Could not open file " << STATS_FILE_NAME << std::endl;
    }
}