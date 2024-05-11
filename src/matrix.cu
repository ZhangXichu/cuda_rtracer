#include <matrix.cuh>

__host__ __device__ uchar3& Matrix::at(int row, int col) 
{
    int index = row * width + col;
    return data[index];
}