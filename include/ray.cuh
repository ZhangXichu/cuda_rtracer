#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>
#include <vector.cuh>

class Ray {

public:

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point origin, const Vector& direction)
           : _origin(origin),
             _direction(direction)
        {}

    __host__ __device__ const Point& origin() const { return _origin; }
    __host__ __device__ const Vector& direction() const { return _direction; }

    __host__ __device__ Point at(double t) const 
    {
        return _origin + t*_direction;
    }
    
private:
    Point _origin;
    Vector _direction;
};

#endif