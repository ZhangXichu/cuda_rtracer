#ifndef HITTABLE_H
#define HITTABLE_H

#include <vector.cuh>
#include <ray.cuh>

class HitRecord {

public:
    Point p;
    Vector normal;
    double t;
    bool front_face;

    __device__ inline void set_face_normal(const Ray& ray, const Vector& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {

public:
     __device__ virtual ~Hittable() {};
     __device__ virtual bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& record) const = 0;
};

#endif