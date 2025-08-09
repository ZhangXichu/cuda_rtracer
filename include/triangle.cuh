#ifndef TRIANGLE_H
#define TRIANGLE_H


#include <hittable.cuh>
#include <vector.cuh>


class Triangle : public Hittable {

public:
__device__ Triangle(const Point& v0, const Point& v1, const Point& v2, Material* m)
        : _v0(v0), _v1(v1), _v2(v2), _material(m)
    {
        _e1 = _v1 - _v0;
        _e2 = _v2 - _v0;
        // Precompute (unnormalized) normal; we'll normalize when setting face normal.
        _n  = cross(_e1, _e2);
    }

__device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override;

private:
Point   _v0, _v1, _v2;
Vector  _e1, _e2;
Vector  _n;           // e1 × e2 (not unit length)    
Material* _material;
};

#endif // TRIANGLE_H