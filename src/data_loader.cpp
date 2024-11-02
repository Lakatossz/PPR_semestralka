#include "data_loader.h"

vector<string> split(const string& s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

vector<vector<double>> read(const string& file_name, const size_t number_of_lines)
{
    ifstream file(file_name);
    
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return {};
    }

    string line;
    char delimiter = ',';  // CSV delimiter
    vector<double> column1_data;
    vector<double> column2_data;
    vector<double> column3_data;

    // Skip header
    getline(file, line);


    for (int i = 1; i < number_of_lines; ++i) {
        getline(file, line);
        // Split the line by delimiter
        vector<string> columns = split(line, delimiter);

        // Make sure the row has enough columns
        if (columns.size() > 3) {

            try {
                column1_data.push_back(std::stod(columns[1]));  // Second column (index 0)
                column2_data.push_back(std::stod(columns[2]));  // Third column (index 0)
                column3_data.push_back(std::stod(columns[3]));  // Fourth column (index 2)
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
            }
        }
    }

    // Read each line from the file
    /*while (getline(file, line)) {
        // Split the line by delimiter
        vector<string> columns = split(line, delimiter);

        // Make sure the row has enough columns
        if (columns.size() > 3) {

            column1_data.push_back(columns[1]);  // Second column (index 0)
            column2_data.push_back(columns[2]);  // Third column (index 0)
            column3_data.push_back(columns[3]);  // Fourth column (index 2)
        }
    } */

    file.close();

    vector<vector<double>> columns;
    columns.push_back(column1_data);
    columns.push_back(column2_data);
    columns.push_back(column3_data);

    return columns;
}