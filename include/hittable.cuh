#ifndef HITTABLE_H
#define HITTABLE_H

#include <vector.cuh>
#include <ray.cuh>

class HitRecord {

public:
    Point p;
    Vector normal;
    double t;
};

class Hittable {

public:
     __host__ __device__ virtual ~Hittable() = default;

      __host__ __device__ bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& record) const = 0;

};

#endif