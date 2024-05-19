#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <matrix.cuh>
#include <vector.cuh>
#include <ray.cuh>
#include <hittable.cuh>
#include <hittable_list.cuh>
#include <rt_weekend.cuh>
#include <sphere.cuh>

struct SceneInfo {
    Vector pixel00_loc;
    Vector camera_center;
    Vector pixel_delta_u;
    Vector pixel_delta_v;
};

__device__ double hit_sphere(const Point& center, double radius, const Ray& ray) {
    Vector oc = center - ray.origin();
    auto a = ray.direction().length_squared();
    auto h = dot(ray.direction(), oc);
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = h*h - a*c;
    if (discriminant < 0)
    {
        return -1.0;
    }
    return (h - sqrt(discriminant)) / a;
}

__device__ Color ray_color(const Ray& ray)
{
    HitRecord record;
    Sphere sphere(Point(0, 0, -1), 0.5);
    Sphere sphere2(Point(0,-100.5,-1), 100);

    if (sphere.hit(ray, 0, infinity, record) || sphere2.hit(ray, 0, infinity, record)) {
        return 0.5 * (record.normal + Color(1, 1, 1));
    }

    Vector unit_direction = unit_vector(ray.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*Color(1.0, 1.0, 1.0) + a*Color(0.5, 0.7, 1.0);
}

__global__ void write_img(Matrix d_img, SceneInfo scene_info)
{
    Vector pixel00_loc = scene_info.pixel00_loc;
    Vector camera_center = scene_info.camera_center;
    Vector pixel_delta_u = scene_info.pixel_delta_u;
    Vector pixel_delta_v = scene_info.pixel_delta_v;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d_img.height && col < d_img.width)
    {
        auto pixel_center = pixel00_loc + (col * pixel_delta_u) + (row * pixel_delta_v);
        auto ray_direction = pixel_center - camera_center;

        Ray ray(camera_center, ray_direction);
        Color pixel_color = ray_color(ray);

        double r = pixel_color.x();
        double g = pixel_color.y();
        double b = pixel_color.z();

        d_img.at(row, col).x = static_cast<unsigned char>(255.999 * r);
        d_img.at(row, col).y = static_cast<unsigned char>(255.999 * g);
        d_img.at(row, col).z = static_cast<unsigned char>(255.999 * b);
    }
}

int main()
{
    auto aspect_ratio = 16.0 / 9.0;
    int img_width = 800;

    int img_height = int(img_width / aspect_ratio);
    img_height = (img_height < 1) ? 1 : img_height;

    // Sphere* sphere1 = new Sphere(Point(0,0,-1), 0.5);
    // Sphere* sphere2 = new Sphere(Point(0,-100.5,-1), 100);

    // camera
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(img_width)/img_height);
    auto camera_center = Point(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = Vector(viewport_width, 0, 0);
    auto viewport_v = Vector(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / img_width;
    auto pixel_delta_v = viewport_v / img_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - Vector(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    const int block_size = 16;

    cudaError_t error;

    size_t size = img_width * img_height * sizeof(uchar3);

    Matrix d_img(img_width, img_height), h_img(img_width, img_height);
    
    h_img.data = new uchar3[size];

    error = cudaMalloc(&d_img.data, size);

    if (error != cudaSuccess) 
    {
        std::cerr << "error while allocating memory for img: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((img_width + dim_block.x-1) / block_size , (img_height + dim_block.y-1) / block_size ); 

    SceneInfo scene_info{pixel00_loc, camera_center, pixel_delta_u, pixel_delta_v};

    write_img<<<dim_grid, dim_block>>>(d_img, scene_info);
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) 
    {
        std::cerr << "cudaDeviceSynchronize error after kernel launch: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    error = cudaMemcpy(h_img.data, d_img.data, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) 
    {
        std::cerr << "cudaMemcpy error during cudaMemcpy: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::ofstream ofs("../output/output.ppm", std::ios::out | std::ios::binary);
    ofs << "P3\n" << img_width << ' ' << img_height << "\n255\n";
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            ofs << (int)h_img.at(i, j).x << ' '
                      << (int)h_img.at(i, j).y << ' '
                      << (int)h_img.at(i, j).z << '\n';
        }
    }

    cudaFree(d_img.data);
    delete[] h_img.data;

    return 0;
}
