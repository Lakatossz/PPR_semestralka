#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdarg> 

using namespace std;

vector<string> split(const string& s, char delimiter);

vector<vector<double>> read(const string& file_name, const size_t number_of_lines);

int countLines(const std::string& filename);