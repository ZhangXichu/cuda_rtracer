#ifndef MATERIAL_H
#define MATERIAL_H

#include <hittable.cuh>
#include <ray.cuh>

class HitRecord;

class Material {

public:
    virtual ~Material() = default;

    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state
    ) const {
        return false;
    }
};

class Lambertian : public Material 
{

public:
    __device__ Lambertian(const Color& albedo) : _albedo(albedo) {}
    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const override;


private:
    Color _albedo;

};

class Metal : public Material 
{

public:

__device__ Metal(const Color& albedo, double fuzz) : _albedo(albedo), _fuzz(fuzz < 1 ? fuzz : 1) {}
__device__ bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const override;


private:
    Color _albedo;
    double _fuzz;

};

class Dielectric : public Material
{

public:

__device__ Dielectric(double refraction_index) : _refraction_index(refraction_index) {}
__device__ bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const override;


private:

double _refraction_index;

};


#endif