#include <iostream>
#include <hittable.cuh>


class HittableList : public Hittable {

public:
    Hittable** objects;
    int size;
    int capacity;

    __device__ HittableList() 
        : objects(nullptr)
        , size(2)
        , capacity(0){}

    __device__ ~HittableList() {
        for (int i = 0; i < size; i++) {
        cudaFree(objects[i]);
        }
        cudaFree(objects);
    }

    __device__ void add(Hittable* object, size_t obj_size);
    __device__ virtual bool hit(const Ray& ray, double ray_tmin, double ray_tmax, HitRecord& record) const override;

};