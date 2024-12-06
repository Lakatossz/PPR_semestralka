#pragma once

#include <chrono>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>
#include <memory>

#define STATS_FILE_NAME "../output/stats.csv"
#define STATS_HEADER "CalcType; Average Duration (s); Min Duration (s); Max Duration (s); Std Dev of Duration (s); Throughput (operations/s); Average Memory Usage (bytes); Max Memory Usage (bytes); Average Thread Count\n"

struct checkPoint {
    double checkPointDuration;
    std::string checkPointName;
};

class PerformanceStats {
public:
    // Record timing statistics
    void startTimer();
    
    void stopTimer(const std::string checkPointName = "checkpoint");

    void printHeadSummary() const;

    // Print summary
    void printSummary(const std::string& calcType, const bool sLabelem = false) const;

    void clearTimer();

    void clearFile();

private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::vector<checkPoint> durations;       // Time in seconds
};