#include "../../include/utils/graph_printer.h"

void writeSVG(const std::string& filename) {
    // Create an output file stream
    std::ofstream svgFile(filename);
    
    // Check if the file is opened successfully
    if (!svgFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write the SVG header
    svgFile << "<svg width=\"400\" height=\"400\" xmlns=\"http://www.w3.org/2000/svg\">\n";

    // Write a rectangle
    svgFile << "  <rect x=\"50\" y=\"50\" width=\"100\" height=\"100\" fill=\"blue\" />\n";

    // Write a circle
    svgFile << "  <circle cx=\"200\" cy=\"200\" r=\"50\" fill=\"red\" />\n";

    // Write a line
    svgFile << "  <line x1=\"10\" y1=\"10\" x2=\"300\" y2=\"300\" stroke=\"green\" stroke-width=\"2\" />\n";

    // Write the SVG footer
    svgFile << "</svg>\n";

    // Close the file
    svgFile.close();
}