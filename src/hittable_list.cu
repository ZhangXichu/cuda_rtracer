#include <hittable_list.cuh>

__device__ void HittableList::add(Hittable* object, size_t obj_size) {
        if (size == capacity) {
            cudaMalloc(&objects, size * sizeof(Hittable*));
            for (int i = 0; i < size; i++) {
                cudaMalloc(&objects[i], obj_size);
            }
        }
        objects[size++] = object;
    }


__device__ bool HittableList::hit(const Ray& ray, double ray_tmin, double ray_tmax, HitRecord& record) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        printf("HittableList::hit called \n");

        for (int i = 0; i < size; i++) {

            printf("hit object i = %d\n", i);

            if (objects[i]->hit(ray, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                record = temp_rec;
            }
        }

        return hit_anything;
    }
