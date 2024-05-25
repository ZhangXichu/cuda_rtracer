#ifndef INTERVAL_H
#define INTERVAL_H

#include <cuda_runtime.h>
#include <rt_weekend.cuh>

class Interval {

public:
    double min, max;

    // __device__ __host__ Interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __device__ __host__ Interval() {}

    __device__ __host__ Interval(double min, double max) : min(min), max(max) {}

    __device__ double size() const {
        return max - min;
    }

    __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }

    __device__ double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
};

extern __constant__ Interval empty;
extern __constant__ Interval universe;

#endif