#include <material.cuh>

__device__ bool Lambertian::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const 
{
    auto scatter_direction = rec.normal + random_unit_vector(state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
    {
        scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction);
    attenuation = _albedo;
    return true;
}

__device__ bool Metal::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const
{
    Vector reflected = reflect(r_in.direction(), rec.normal);
    scattered = Ray(rec.p, reflected);
    attenuation = _albedo;
    return true;
}