#ifndef HITTABLE_H
#define HITTABLE_H

#include <vector.cuh>
#include <ray.cuh>
#include <interval.cuh>
#include <material.cuh>

class Material;

class HitRecord {

public:
    Point p;
    Vector normal;
    Material *material;
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
     __device__ virtual bool hit(const Ray& r, Interval ray_t, HitRecord& record) const = 0;
};

#endif