#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <rt_weekend.cuh>
#include <cuda_runtime.h>

class Vector {

public:

__host__ __device__ Vector() {}

__host__ __device__ Vector(double e0, double e1, double e2)
{
    e.x = e0;
    e.y = e1;
    e.z = e2;
}

__host__ __device__ double x() const { return e.x; }
__host__ __device__ double y() const { return e.y; }
__host__ __device__ double z() const { return e.z; }

__host__ __device__ double& x() { return e.x; }
__host__ __device__ double& y() { return e.y; }
__host__ __device__ double& z() { return e.z; }

__host__ __device__ Vector operator-() const { return Vector(-e.x, -e.y, -e.z); }

__host__ __device__ Vector& operator+=(const Vector& v)
{
    e.x += v.x();
    e.y += v.y();
    e.z += v.z();

    return *this;
}

__host__ __device__ Vector& operator*=(double t)
{
    e.x *= t;
    e.y *= t;
    e.z *= t;

    return *this;
}

__host__ __device__ Vector& operator/=(double t)
{
    return *this *= 1/t;
}

__host__ __device__ double length_squared() const
{
    return e.x*e.x + e.y*e.y + e.z*e.z;
}

__host__ __device__ double length() const 
{
    return sqrt(length_squared());
}

__host__ __device__ bool near_zero() const 
{
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}

private:

    double3 e;

};

using Point = Vector;

__host__ __device__ inline Vector operator+(const Vector& u, const Vector& v)
{
    return Vector(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ inline Vector operator-(const Vector& u, const Vector& v)
{
    return Vector(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__device__ inline Vector operator*(const Vector& u, const Vector& v)
{
    return Vector(u.x()*v.x(), u.y()*v.y(), u.z()*v.z());
}

__host__ __device__ inline Vector operator*(double t, const Vector& v)
{
    return Vector(t*v.x(), t*v.y(), t*v.z());
}

__host__ __device__ inline Vector operator*(const Vector& v, double t)
{
    return t*v;
}

__host__ __device__ inline Vector operator/(const Vector& v, double t)
{
    return (1/t)*v;
}

__host__ __device__ inline double dot(const Vector& u, const Vector& v)
{
    return u.x()*v.x() + u.y()*v.y() + u.z()*v.z();
}

__host__ __device__ inline Vector cross(const Vector& u, const Vector& v)
{
    return Vector(u.y()*v.z() - u.z()*v.y(),
                  u.z()*v.x() - u.x()*v.z(),
                  u.x()*v.y() - u.y()*v.x());
}

__host__ __device__ inline Vector unit_vector(const Vector& v)
{
    return v / v.length();
}

__device__ inline Vector random_vec(curandState *state) 
{
    return Vector(random_double(state), random_double(state), random_double(state));
}

__device__ inline Vector random_vec(curandState *state, double min, double max) 
{
    return Vector(random_double(state, min, max), random_double(state, min, max), random_double(state, min, max));
}

__device__ inline Vector random_in_unit_sphere(curandState *state) {
    while (true) {
        auto p = random_vec(state, -1,1);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ inline Vector random_unit_vector(curandState *state) {
    return unit_vector(random_in_unit_sphere(state));
}

__device__ inline Vector random_on_hemisphere(curandState *state, const Vector& normal) {
    Vector on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline Vector reflect(const Vector& v, const Vector& n) {
    return v - 2*dot(v,n)*n;
}

using Color = Vector;

#endif
