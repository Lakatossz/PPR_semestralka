#include "../../include/sort/bitonic_sort.h"

void padToPowerOfTwo(std::vector<double>& arr) {
    size_t originalSize = arr.size();
    size_t nextPowerOf2 = pow(2, ceil(log2(originalSize)));

    if (originalSize < nextPowerOf2) {
        arr.resize(nextPowerOf2, std::numeric_limits<double>::max()); // Pad with max value or any other placeholder
    }
}

// Helper function to remove padding after sorting
void removePadding(std::vector<double>& arr, size_t originalSize) {
    arr.resize(originalSize);
}

// Helper function to swap elements if needed, depending on the sort order
inline void bitonicCompare(std::vector<double>& arr, int i, int j, bool ascending) {
    if (ascending == (arr[i] > arr[j])) {
        std::cout << arr[i] << " " << arr[j] << std::endl;
        std::swap(arr[i], arr[j]);
    }
}


// Function to process one "merge chunk"
void processChunk(std::vector<double>& arr, int low, int count, int step, bool ascending) {
    for (int i = low; i < low + count - step; i++) {
        std::cout << (i + step) << " < " << (low + count) << std::endl;
        if (i + step < low + count) {
            bitonicCompare(arr, i, i + step, ascending);
        }
    }
}

// Main merge function with chunked processing
void bitonicMerge(std::vector<double>& arr, int low, int count, bool ascending) {
    unsigned int num_threads = std::thread::hardware_concurrency();  // Get available threads
    size_t num_chunks = num_threads;  // Divide into chunks based on available threads
    size_t chunk_size = count / num_chunks;

    // Create a range to iterate over the chunks
    std::vector<size_t> chunk_indices(num_chunks);
    std::iota(chunk_indices.begin(), chunk_indices.end(), 0);  // Fill with 0, 1, ..., num_chunks-1

    // Iterate over chunks, using std::for_each to process them in parallel
    std::for_each(std::execution::seq, chunk_indices.begin(), chunk_indices.end(), 
        [&arr, low, chunk_size, ascending, count, num_chunks](size_t i) {
            size_t chunk_start = low + i * chunk_size;
            size_t chunk_end = (i == num_chunks - 1) ? (low + count) : (chunk_start + chunk_size);

            for (int step = chunk_size / 2; step > 0; step /= 2) {
                processChunk(arr, chunk_start, chunk_end - chunk_start, step, ascending);
            }
        });
}

// Recursive function to produce a bitonic sequence and sort it
void bitonicSortRecursive(std::vector<double>& arr, int low, int count, bool ascending) {
    if (count > 1) {
        int k = count / 2;

        // Create ascending and descending subsequences
        bitonicSortRecursive(arr, low, k, true);
        bitonicSortRecursive(arr, low + k, k, false);

        // Merge the sequences
        bitonicMerge(arr, low, count, ascending);
    }
}

// Function to initiate bitonic sort
void bitonicSort(std::vector<double>& arr, bool ascending) {
    int n = arr.size();

    padToPowerOfTwo(arr);

    // Perform the recursive sort
    bitonicSortRecursive(arr, 0, n, ascending);

    removePadding(arr, n);
}

// Function to perform a bitonic sort on a vector of four double values
void bitonic_sort(__m256d& vec) {
    // Create an array to hold the sorted results temporarily
    double temp[4];
    
    // Store the vector in an array
    _mm256_storeu_pd(temp, vec);
    
    // Compare and swap
    // 1. Sort the first half in ascending order and the second half in descending order
    if (temp[0] > temp[1]) std::swap(temp[0], temp[1]);
    if (temp[2] > temp[3]) std::swap(temp[2], temp[3]);
    if (temp[0] > temp[2]) std::swap(temp[0], temp[2]);
    if (temp[1] > temp[3]) std::swap(temp[1], temp[3]);
    if (temp[1] > temp[2]) std::swap(temp[1], temp[2]);

    // Load the sorted values back into the vector
    vec = _mm256_loadu_pd(temp);
}

void sort(std::vector<double> data, size_t size_) {
    size_t size = data.size();
    
    // Sort each chunk of 4 doubles
    for (size_t i = 0; i < size / 4 * 4; i += 4) {
        __m256d vec = _mm256_loadu_pd(&data[i]);
        bitonic_sort(vec);
        _mm256_storeu_pd(&data[i], vec);
    }

    // Handle remaining elements if size is not a multiple of 4
    for (size_t i = (size / 4) * 4; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j) {
            if (data[i] > data[j]) {
                std::swap(data[i], data[j]);
            }
        }
    }

    // Merge sorted chunks
    std::vector<double> merged(size);
    size_t chunk_size = 4;

    for (size_t i = 0; i < size; i += chunk_size * 2) {
        size_t left_end = std::min(i + chunk_size, size);
        size_t right_end = std::min(i + chunk_size * 2, size);

        std::vector<double> left(data.begin() + i, data.begin() + left_end);
        std::vector<double> right(data.begin() + left_end, data.begin() + right_end);

        size_t left_idx = 0, right_idx = 0, merged_idx = i;

        // Merge left and right arrays
        while (left_idx < left.size() && right_idx < right.size()) {
            if (left[left_idx] <= right[right_idx]) {
                merged[merged_idx++] = left[left_idx++];
            } else {
                merged[merged_idx++] = right[right_idx++];
            }
        }

        // Copy remaining elements
        while (left_idx < left.size()) {
            merged[merged_idx++] = left[left_idx++];
        }
        while (right_idx < right.size()) {
            merged[merged_idx++] = right[right_idx++];
        }
    }

    // Copy merged result back to data
    data = std::move(merged);
}

// Function to sort an array of doubles using AVX2
/*void sort(std::vector<double> data, size_t size) {
    // Process the vector in chunks of 4
    for (size_t i = 0; i < size; i += 4) {
        // Load 4 double values from the array, checking to avoid out-of-bounds
        __m256d vec = _mm256_loadu_pd(&data[i]);
        // Sort the vector
        bitonic_sort(vec);
        // Store the sorted values back to the array
        _mm256_storeu_pd(&data[i], vec);
    }
}*/

// Function to perform AVX2 bitonic merge on a 4-element vector
void bitonic_merge_avx2(__m256d& vec, bool ascending) {
    double temp[4];
    _mm256_storeu_pd(temp, vec);

    // Perform bitonic merge: compare and swap in pairs
    if ((temp[0] > temp[1]) == ascending) std::swap(temp[0], temp[1]);
    if ((temp[2] > temp[3]) == ascending) std::swap(temp[2], temp[3]);
    if ((temp[0] > temp[2]) == ascending) std::swap(temp[0], temp[2]);
    if ((temp[1] > temp[3]) == ascending) std::swap(temp[1], temp[3]);
    if ((temp[1] > temp[2]) == ascending) std::swap(temp[1], temp[2]);

    // Store back sorted values into the vector
    vec = _mm256_loadu_pd(temp);
}

// Function to recursively perform bitonic merge over an entire array using AVX2
void bitonic_merge_recursive(std::vector<double> data, size_t start, size_t length, bool ascending) {
    if (length <= 4) {
        // For small chunks of 4 or fewer, use AVX2 directly
        __m256d vec = _mm256_loadu_pd(&data[start]);
        bitonic_merge_avx2(vec, ascending);
        _mm256_storeu_pd(&data[start], vec);
    } else {
        size_t half = length / 2;
        for (size_t i = start; i < start + half; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(&data[i]);
            __m256d vec2 = _mm256_loadu_pd(&data[i + half]);

            // Perform min/max comparisons for merging
            __m256d min_vals = _mm256_min_pd(vec1, vec2);
            __m256d max_vals = _mm256_max_pd(vec1, vec2);

            // Store in ascending or descending order based on the merge direction
            if (ascending) {
                _mm256_storeu_pd(&data[i], min_vals);
                _mm256_storeu_pd(&data[i + half], max_vals);
            } else {
                _mm256_storeu_pd(&data[i], max_vals);
                _mm256_storeu_pd(&data[i + half], min_vals);
            }
        }
        // Recursively merge the halves
        bitonic_merge_recursive(data, start, half, ascending);
        bitonic_merge_recursive(data, start + half, half, ascending);
    }
}

// Bitonic sort function using AVX2
void bitonic_sort_avx2(std::vector<double> data, size_t start, size_t length, bool ascending) {
    if (length <= 4) {
        // Base case: if we reach 4 or fewer elements, perform AVX2 bitonic sort directly
        __m256d vec = _mm256_loadu_pd(&data[start]);
        bitonic_merge_avx2(vec, ascending);
        _mm256_storeu_pd(&data[start], vec);
    } else {
        size_t half = length / 2;
        // Sort each half in opposite directions to create a bitonic sequence
        bitonic_sort_avx2(data, start, half, true);       // Sort the first half in ascending order
        bitonic_sort_avx2(data, start + half, half, false); // Sort the second half in descending order

        // Merge the entire sequence into a single sorted sequence
        bitonic_merge_recursive(data, start, length, ascending);
    }
}
