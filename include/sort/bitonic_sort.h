#pragma once

#include <immintrin.h>
#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

void padToPowerOfTwo(std::vector<double>& arr);

// Helper function to remove padding after sorting
void removePadding(std::vector<double>& arr, size_t originalSize);

// Helper function to swap elements if needed, depending on the sort order
void bitonicCompare(std::vector<double>& arr, int i, int j, bool ascending);

// The main bitonic merge function
void bitonicMerge(std::vector<double>& arr, int low, int count, bool ascending);

// Recursive function to produce a bitonic sequence and sort it
void bitonicSortRecursive(std::vector<double>& arr, int low, int count, bool ascending);

// Function to initiate bitonic sort
void bitonicSort(std::vector<double>& arr, bool ascending);

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

