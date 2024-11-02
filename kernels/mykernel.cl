__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

__kernel void compute_mean(__global const float* data, __global float* mean, int N) {
    int id = get_global_id(0);
    mean[id] = data[id] / N;
}

__kernel void compute_variance(__global const float* data, __global const float* mean, __global float* variance, int N) {
    int id = get_global_id(0);
    float diff = data[id] - mean[0];
    variance[id] = (diff * diff) / N;
}

__kernel void calculate_absolute_deviation(
    __global const double* data, 
    __global double* deviations, 
    const double median) 
{
    int i = get_global_id(0);
    deviations[i] = fabs(data[i] - median);
}