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
#include <interval.cuh>

struct SceneInfo {
    Vector pixel00_loc;
    Vector camera_center;
    Vector pixel_delta_u;
    Vector pixel_delta_v;
};

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

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

__device__ Color ray_color(curandState* rand_states, int max_depth, const Ray& ray)
{
    Color accumulated_color(1.0, 1.0, 1.0); 
    Ray current_ray = ray;
    int depth = 0;

    HitRecord record;

    Sphere sphere(Point(0, 0, -1), 0.5);
    Sphere sphere2(Point(0,-100.5,-1), 100);

    while (depth < max_depth) {
        
        if (sphere.hit(current_ray, Interval(0.0, infinity), record) || sphere2.hit(current_ray, Interval(0.0, infinity), record)) 
        {
            Vector direction = random_on_hemisphere(rand_states, record.normal);
            current_ray = Ray(record.p, direction);
            accumulated_color *= 0.5;
            
        } else {
            
            Vector unit_direction = unit_vector(current_ray.direction());
            auto a = 0.5*(unit_direction.y() + 1.0);
            Color background = (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);

            accumulated_color = accumulated_color * background;
            break;
        }
        
        depth++;
    }
    return accumulated_color;
}

__device__ Vector sample_square(curandState* rand_states) {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
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

__global__ void write_img(Matrix d_img, SceneInfo scene_info, int samples_per_pixel, curandState* rand_states)
{
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
            pixel_color += ray_color(rand_states, 50, ray);
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

    auto aspect_ratio = 16.0 / 9.0;
    int img_width = 800;
    int samples_per_pixel = 50; 

    int img_height = int(img_width / aspect_ratio);
    img_height = (img_height < 1) ? 1 : img_height;

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

    curandState* rand_states;
    int N = 1024;
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    cudaMalloc((void**)&rand_states, N * sizeof(curandState));

    setup_kernel<<<num_blocks, num_threads>>>(rand_states, time(0));

    write_img<<<dim_grid, dim_block>>>(d_img, scene_info, samples_per_pixel, rand_states);

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
    cudaFree(rand_states);
    delete[] h_img.data;

    return 0;
}
