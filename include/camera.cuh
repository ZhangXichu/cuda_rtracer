#ifndef CMAERA_H
#define CAMERA_H

#include <vector.cuh>
#include <ray.cuh>
#include <hittable.cuh>
#include <sphere.cuh>

extern __device__ Hittable** sphere_lst;
extern __device__ Hittable* world;

struct SceneInfo {
    Vector pixel00_loc;
    Vector camera_center;
    Vector pixel_delta_u;
    Vector pixel_delta_v;
};


class Camera {

public:
    double aspect_ratio = 16.0 / 9.0;
    int img_width = 800;
    double vfov = 90;  // Vertical view angle (field of view)
    Point lookfrom = Point(0,0,0);   // Point camera is looking from
    Point lookat   = Point(0,0,-1);  // Point camera is looking at
    Vector   vup   = Vector(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    int get_img_height() const;
    __host__ __device__ SceneInfo get_scene_info() const;
    __host__ __device__ void initialize();
    __device__ Color ray_color(curandState* rand_states, int max_depth, const Ray& ray);
    __device__ Vector sample_square(curandState* rand_states);
    __device__ Point defocus_disk_sample(curandState* rand_states) const;
    __device__ Ray get_ray(int i, int j, curandState* rand_states);


private:
    int _img_height;
    Point _camera_center;
    Point _pixel00_loc;
    Vector _pixel_delta_u; // Offset to pixel to the right
    Vector _pixel_delta_v; // Offset to pixel below
    SceneInfo _scene_info;
    Vector   _u, _v, _w;   // Camera frame basis vectors
    Vector   _defocus_disk_u;       // Defocus disk horizontal radius
    Vector   _defocus_disk_v;       // Defocus disk vertical radius
};


#endif
