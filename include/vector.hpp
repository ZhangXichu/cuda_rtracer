#include <cuda_runtime.h>

class Vector {

public:

double e[3];

__host__ __device__ Vector(): e{0, 0, 0} {}


};