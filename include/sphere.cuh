#ifndef SPHERE_H
#define SPHERE_H


#include <hittable.cuh>

__device__ __forceinline__ double my_fmax(double a, double b) {
    return a > b ? a : b;
}

class Sphere : public Hittable {

public:
    // Stationary Sphere
    __device__ Sphere(const Point& center, double radius, Material* material)
        : _center1(center)
        , _radius(my_fmax(0, radius))
        , _material(material)
        , _is_moving(false) {}

    __device__ Sphere(const Point& center, const Point& center2, double radius, Material* material)
        : _center1(center)
        , _radius(my_fmax(0, radius))
        , _material(material) 
        , _is_moving(true)
        {
            _center_vec = center2 - _center1;
        }

    __device__ ~Sphere() {}
    __device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const override;
 
private:
    Point _center1;
    double _radius;
    Material* _material;
    bool _is_moving;
    Vector _center_vec;

    __device__ Point sphere_center(double time) const;
};


#endif