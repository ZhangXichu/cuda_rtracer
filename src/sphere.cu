#include <sphere.cuh>

__device__ bool Sphere::hit(const Ray& ray, Interval ray_t, HitRecord& record) const 
{
    Vector oc = _center - ray.origin();
    auto a = ray.direction().length_squared();
    auto h = dot(ray.direction(), oc);
    auto c = oc.length_squared() - _radius*_radius;

    auto discriminant = h*h - a*c;
    if (discriminant < 0)
        return false;

    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    record.t = root;
    record.p = ray.at(record.t);
    Vector outward_normal = (record.p - _center) / _radius;
    record.set_face_normal(ray, outward_normal);

    return true;
}