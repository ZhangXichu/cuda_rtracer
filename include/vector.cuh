#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
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

__host__ __device__ inline Vector operator*(const Vector& u, const Vector& v)
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

using Color = Vector;

#endif
