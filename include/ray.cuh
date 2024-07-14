#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>
#include <vector.cuh>

class Ray {

public:

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point origin, const Vector& direction)
           : _origin(origin),
             _direction(direction),
             _time(0)
        {}
    __host__ __device__ Ray(const Point origin, const Vector& direction, double time)
           : _origin(origin),
             _direction(direction),
             _time(time)
        {}

    __host__ __device__ const Point& origin() const { return _origin; }
    __host__ __device__ const Vector& direction() const { return _direction; }

    __host__ __device__ Point at(double t) const 
    {
        return _origin + t*_direction;
    }

    __host__ __device__ double time() const { return _time; }
    
private:
    Point _origin;
    Vector _direction;
    double _time;
};

#endif