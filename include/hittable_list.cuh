#include <iostream>
#include <hittable.cuh>


class HittableList : public Hittable {

public:
    Hittable** objects;
    int size;
    int capacity;

    __device__ HittableList(Hittable** lst_objs, int lst_size) 
        : objects(lst_objs)
        , size(lst_size){}

    __device__ ~HittableList() {
        for (int i = 0; i < size; i++) {
        cudaFree(objects[i]);
        }
        cudaFree(objects);
    }

    __device__ void add(Hittable* object, size_t obj_size);
    __device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const override;

};