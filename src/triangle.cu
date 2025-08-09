#include <triangle.cuh>

__device__ bool Triangle::hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        // Möller–Trumbore intersection
        const Vector pvec = cross(r.direction(), _e2);
        const double det  = dot(_e1, pvec);

        // For two-sided triangles keep |det| check; for backface culling use (det <= EPS).
        const double EPS = 1e-12;
        if (fabs(det) < EPS) return false;

        const double invDet = 1.0 / det;

        const Vector tvec = r.origin() - _v0;
        const double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        const Vector qvec = cross(tvec, _e1);
        const double v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0 || (u + v) > 1.0) return false;

        const double t = dot(_e2, qvec) * invDet;
        if (!ray_t.surrounds(t)) return false;

        rec.t = t;
        rec.p = r.at(t);

        // Use the geometric normal; flip based on ray to keep consistency with current API.
        const Vector outward = unit_vector(_n);
        rec.set_face_normal(r, outward);

        rec.material = _material;
        return true;
    }