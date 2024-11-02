#pragma once

#include <immintrin.h>
#include <iostream>
#include <array>
#include <vector>

// Function to perform a bitonic sort on a vector of four double values
void bitonic_sort(__m256d& vec);

// Function to sort an array of doubles using AVX2
void sort(std::vector<double> data, size_t size_);

// Function to perform AVX2 bitonic merge on a 4-element vector
void bitonic_merge_avx2(__m256d& vec, bool ascending);

// Function to recursively perform bitonic merge over an entire array using AVX2
void bitonic_merge_recursive(std::vector<double> data, size_t start, size_t length, bool ascending);

// Bitonic sort function using AVX2
void bitonic_sort_avx2(std::vector<double> data, size_t start, size_t length, bool ascending);

