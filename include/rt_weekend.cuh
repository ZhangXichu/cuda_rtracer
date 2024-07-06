#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <curand_kernel.h>
// #include <random>


// Constants

__device__ const double infinity = std::numeric_limits<double>::infinity();
__device__ const double pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double random_double(curandState *state) {
    // Returns a random real in [0,1).
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    return curand_uniform(&state[id]);
}

__device__ inline double random_double(curandState* state, double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(state);
}

#endif