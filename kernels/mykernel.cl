__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

__kernel void compute_mean(__global const float* data, __global float* mean, int N) {
    int id = get_global_id(0);
    mean[id] = data[id] / N;
}

__kernel void sum_reduction(__global const double* data, __global double* result, const int n) {
    // Allocate shared memory for partial sums
    __local double local_sum[256];  // Assuming a maximum of 256 work-items per work-group

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Each thread loads a value into local memory
    local_sum[local_id] = (global_id < n) ? data[global_id] : 0.0;

    // Synchronize to make sure all threads have loaded their data
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in shared memory
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // The first thread in each work group writes the partial sum to the result array
    if (local_id == 0) {
        result[get_group_id(0)] = local_sum[0];
    }
}

__kernel void compute_variance(__global const double* data, const double mean, __global double* variance, const int n) {
    __local double local_var[256];  // Adjust local size as needed

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Load squared difference into local memory
    double diff = (global_id < n) ? data[global_id] - mean : 0.0;
    local_var[local_id] = diff * diff;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform parallel reduction
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_var[local_id] += local_var[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the partial sum from each work-group to the variance buffer
    if (local_id == 0) {
        variance[get_group_id(0)] = local_var[0];
    }
}

/*__kernel void compute_variance(__global const float* data, __global const float* mean, __global float* variance, int N) {
    int id = get_global_id(0);
    float diff = data[id] - mean[0];
    variance[id] = (diff * diff) / N;
}*/

__kernel void calculate_absolute_deviation(
    __global const double* data, 
    __global double* deviations, 
    const double median) 
{
    int i = get_global_id(0);
    deviations[i] = fabs(data[i] - median);
}