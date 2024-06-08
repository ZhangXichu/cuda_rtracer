#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <matrix.cuh>

#include <hittable_list.cuh>
#include <rt_weekend.cuh>

#include <interval.cuh>
#include <error_check.cuh>
#include <camera.cuh>


// __device__ Hittable** sphere_lst;
// __device__ Hittable* world;

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// __device__ double hit_sphere(const Point& center, double radius, const Ray& ray) {
//     Vector oc = center - ray.origin();
//     auto a = ray.direction().length_squared();
//     auto h = dot(ray.direction(), oc);
//     auto c = oc.length_squared() - radius*radius;
//     auto discriminant = h*h - a*c;
//     if (discriminant < 0)
//     {
//         return -1.0;
//     }
//     return (h - sqrt(discriminant)) / a;
// }

// __device__ Color ray_color(curandState* rand_states, int max_depth, const Ray& ray)
// {
//     Color accumulated_color(1.0, 1.0, 1.0); 
//     Ray current_ray = ray;
//     int depth = 0;

//     HitRecord record;

//     Sphere sphere(Point(0, 0, -1), 0.5);
//     Sphere sphere2(Point(0,-100.5,-1), 100);

//     while (depth < max_depth) {
        
//         if (world->hit(current_ray, Interval(0.0, infinity), record))
//         {
//             Vector direction = random_on_hemisphere(rand_states, record.normal);
//             current_ray = Ray(record.p, direction);
//             accumulated_color *= 0.5;
            
//         } else {
            
//             Vector unit_direction = unit_vector(current_ray.direction());
//             auto a = 0.5*(unit_direction.y() + 1.0);
//             Color background = (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);

//             accumulated_color = accumulated_color * background;
//             break;
//         }
        
//         depth++;
//     }
//     return accumulated_color;
// }

__device__ Vector sample_square(curandState* rand_states) {
        return Vector(random_double(rand_states) - 0.5, random_double(rand_states) - 0.5, 0);
    }

__device__ Ray get_ray(int i, int j, SceneInfo scene_info, curandState* rand_states)
{
    // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.
    auto offset = sample_square(rand_states);
    auto pixel_sample = scene_info.pixel00_loc
                        + ((i + offset.x()) * scene_info.pixel_delta_u)
                        + ((j + offset.y()) * scene_info.pixel_delta_v);

    auto ray_origin = scene_info.camera_center;
    auto ray_direction = pixel_sample - ray_origin;

    return Ray(ray_origin, ray_direction);
}


// reference https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
// section 7.34
__global__ void create_world()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        sphere_lst = (Hittable**)malloc(2 * sizeof(Hittable*));
        world = (Hittable*)malloc(sizeof(Hittable*));

        printf("The memory address of sphere_lst is: %p\n", (void*)&sphere_lst);

        sphere_lst[0] = new Sphere(Point(0, 0, -1), 0.5);
        sphere_lst[1] = new Sphere(Point(0,-100.5,-1), 100);

        printf("The memory address of sphere1 and sphere2 are: %p, %p\n", (void*)sphere_lst[0], (void*)sphere_lst[1]);
        printf("Memory address of world: %p\n", (void*)world);

        world = new HittableList(sphere_lst, 2);
    }

}

__global__ void free_world()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        free(sphere_lst);
    }
}

__global__ void write_img(Matrix d_img, Camera camera, int samples_per_pixel, curandState* rand_states)
{
    SceneInfo scene_info = camera.get_scene_info();

    Vector pixel00_loc = scene_info.pixel00_loc;
    Vector camera_center = scene_info.camera_center;
    Vector pixel_delta_u = scene_info.pixel_delta_u;
    Vector pixel_delta_v = scene_info.pixel_delta_v;

    double pixel_samples_scale = 1.0 / samples_per_pixel;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    const Interval intensity(0.000, 0.999);

    if (row < d_img.height && col < d_img.width)
    {
        auto pixel_center = pixel00_loc + (col * pixel_delta_u) + (row * pixel_delta_v);
        auto ray_direction = pixel_center - camera_center;

        Color pixel_color(0, 0, 0);
        for (int sample = 0; sample < samples_per_pixel; sample++){
            Ray ray = get_ray(col, row, scene_info, rand_states);
            pixel_color += camera.ray_color(rand_states, 50, ray);
        }

        pixel_color = pixel_samples_scale * pixel_color;

        double r = pixel_color.x();
        double g = pixel_color.y();
        double b = pixel_color.z();

        d_img.at(row, col).x = static_cast<unsigned char>(255.999 * intensity.clamp(r));
        d_img.at(row, col).y = static_cast<unsigned char>(255.999 * intensity.clamp(g));
        d_img.at(row, col).z = static_cast<unsigned char>(255.999 * intensity.clamp(b));
    }
}

int main()
{
    // initialize constants empty and universal
    Interval h_empty(+infinity, -infinity);
    Interval h_universe(-infinity, +infinity);
    cudaMemcpyToSymbol(empty, &h_empty, sizeof(Interval));
    cudaMemcpyToSymbol(universe, &h_universe, sizeof(Interval));

    cudaDeviceSetLimit(cudaLimitStackSize, 65536);

    int samples_per_pixel = 50; 

    Camera camera;
    
    camera.aspect_ratio = 16.0 / 9.0;
    camera.img_width = 800;

    camera.initialize();

    const int block_size = 16;

    int img_height = camera.get_img_height();

    SceneInfo scene_info = camera.get_scene_info();

    size_t size = camera.img_width * img_height * sizeof(uchar3);

    Matrix d_img(camera.img_width, img_height), h_img(camera.img_width, img_height);
    
    h_img.data = new uchar3[size];

    GPU_ERR_CHECK(cudaMalloc(&d_img.data, size));

    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((camera.img_width + dim_block.x-1) / block_size , (img_height + dim_block.y-1) / block_size ); 

    // SceneInfo scene_info{pixel00_loc, camera_center, pixel_delta_u, pixel_delta_v};

    curandState* rand_states;
    int N = 1024;
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    cudaMalloc((void**)&rand_states, N * sizeof(curandState));

    setup_kernel<<<num_blocks, num_threads>>>(rand_states, time(0));

    create_world<<<1,1>>>();

    write_img<<<dim_grid, dim_block>>>(d_img, camera, samples_per_pixel, rand_states);

    GPU_ERR_CHECK(cudaDeviceSynchronize());

    GPU_ERR_CHECK(cudaMemcpy(h_img.data, d_img.data, size, cudaMemcpyDeviceToHost));

    std::ofstream ofs("../output/output.ppm", std::ios::out | std::ios::binary);
    ofs << "P3\n" << camera.img_width << ' ' << img_height << "\n255\n";
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < camera.img_width; j++) {
            ofs << (int)h_img.at(i, j).x << ' '
                      << (int)h_img.at(i, j).y << ' '
                      << (int)h_img.at(i, j).z << '\n';
        }
    }

    free_world<<<1,1>>>();

    cudaFree(d_img.data);
    cudaFree(rand_states);
    delete[] h_img.data;

    return 0;
}
