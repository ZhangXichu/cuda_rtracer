#include <sphere.cuh>

__device__ bool Sphere::hit(const Ray& ray, Interval ray_t, HitRecord& record) const 
{
    Point center = _is_moving ? sphere_center(ray.time()) : _center1;
    Vector oc = center - ray.origin();
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
    Vector outward_normal = (record.p - center) / _radius;
    record.set_face_normal(ray, outward_normal);
    record.material = _material;

    return true;
}

__device__ Point Sphere::sphere_center(double time) const
{
    // Linearly interpolate from center1 to center2 according to time, where t=0 yields
        // center1, and t=1 yields center2.
        return _center1 + time*_center_vec;
}