#include <material.cuh>

__device__ bool Lambertian::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const 
{
    auto scatter_direction = rec.normal + random_unit_vector(state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
    {
        scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction, r_in.time());
    attenuation = _albedo;
    return true;
}

__device__ bool Metal::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const
{
    Vector reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (_fuzz * random_unit_vector(state));
    scattered = Ray(rec.p, reflected, r_in.time());
    attenuation = _albedo;
    // return true;
    return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ bool Dielectric::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState *state) const
{
    attenuation = Color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0/_refraction_index) : _refraction_index;

    Vector unit_direction = unit_vector(r_in.direction());

    double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    Vector direction;

    if (cannot_refract || reflectance(cos_theta, ri) > random_double(state))
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, ri);

    scattered = Ray(rec.p, direction, r_in.time());

    // Vector refracted = refract(unit_direction, rec.normal, ri);

    // scattered = Ray(rec.p, refracted);
    
    return true;
}

__device__ double Dielectric::reflectance(double cosine, double refraction_index) const {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }