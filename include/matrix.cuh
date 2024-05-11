#ifndef MATRIX_CUH
#define MATRIX_CUH

struct Matrix {
    int width;
    int height;
    uchar3* data;

    __host__ __device__ Matrix(int w, int h) : 
        width(w), height(h) 
    {}

    __host__ __device__ uchar3& at(int row, int col);
};

#endif