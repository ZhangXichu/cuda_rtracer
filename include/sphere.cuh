#ifndef SPHERE_H
#define SPHERE_H


#include <hittable.cuh>

__device__ __forceinline__ double my_fmax(double a, double b) {
    return a > b ? a : b;
}

class Sphere : public Hittable {

public:
    __device__ Sphere(const Point& center, double radius, Material* material)
        : _center(center)
        , _radius(my_fmax(0, radius))
        , _material(material) {}

    __device__ ~Sphere() {}
    __device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const override;
 
private:
    Point _center;
    double _radius;
    Material* _material;
};


#endif