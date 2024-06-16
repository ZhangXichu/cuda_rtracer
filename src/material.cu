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
    reflected = unit_vector(reflected) + (_fuzz * random_unit_vector(state));
    scattered = Ray(rec.p, reflected);
    attenuation = _albedo;
    // return true;
    return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ bool Dielectric::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const
{
    attenuation = Color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0/_refraction_index) : _refraction_index;

    Vector unit_direction = unit_vector(r_in.direction());
    Vector refracted = refract(unit_direction, rec.normal, ri);

    scattered = Ray(rec.p, refracted);
    
    return true;
}