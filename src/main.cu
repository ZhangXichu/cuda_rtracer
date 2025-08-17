#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <matrix.cuh>

#include <hittable_list.cuh>
#include <rt_weekend.cuh>

#include <interval.cuh>
#include <error_check.cuh>
#include <camera.cuh>
#include <triangle.cuh>
#include <obj_loader.cuh>

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ inline Point vtxf(const float* V, int i) { // fetch vertex
    return Point((double)V[3*i+0], (double)V[3*i+1], (double)V[3*i+2]);
}

// yaw rotation around Y
__device__ inline Point yaw_y(const Point& p, double c, double s) {
    double x =  p.x()*c + p.z()*s;
    double z = -p.x()*s + p.z()*c;
    return Point(x, p.y(), z);
}

// reference https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
// section 7.34
__global__ void create_world(curandState* rand_states,
                             const float* verts, int vcount,
                             const int3* faces, int fcount,
                             double scale,      // S
                             Point translate,   // T
                             double yaw_deg)    // rotation about Y in degrees
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // int num_spheres = 190;

        // obj_lst = (Hittable**)malloc(num_spheres * sizeof(Hittable*));
        // world = (Hittable*)malloc(sizeof(Hittable*));

        // Material* ground = new Lambertian(Color(0.5, 0.5, 0.5));
        // obj_lst[0] = new Sphere(Point(0,-1000,0), 1000, ground);  // ground

        // Material* material1 = new Dielectric(1.5);
        // obj_lst[1] = new Sphere(Point(0, 1, 0), 1, material1);

        // Material* material2 = new Lambertian(Color(0.4, 0.2, 0.1));
        // obj_lst[2] = new Sphere(Point(-4, 1, 0), 1, material2);

        // Material* material3 = new Metal(Color(184.0/225.0, 115.0/225.0, 51.0/225.0), 0);
        // obj_lst[3] = new Sphere(Point(4, 1, 0), 1, material3);

        // int index = 4;

        // for (int a = -7; a < 7; a+=1.5)
        // {
        //     for (int b = -7; b < 7; b+=1.5) 
        //     {
        //         auto choose_mat = random_double(rand_states);

        //         Point center(a + 7.5*random_double(rand_states), 0.2, b + 7.5*random_double(rand_states));

        //         if ((center - Point(4, 0.2, 0)).length() > 0.9) 
        //         {
        //             Material* material;

        //             if (choose_mat < 0.5) 
        //             {
        //                 auto albedo = random_vec(rand_states) * random_vec(rand_states);
        //                 auto center2 = center + Vector(0, random_double(rand_states, 0, 1), 0);
                        
        //                 material = new Lambertian(albedo);
        //                 // obj_lst[index] = new Sphere(center, center2, 0.2, material);
        //                 obj_lst[index] = new Sphere(center, 0.2, material);
        //                 index += 1;
        //             } else if (choose_mat < 0.8) 
        //             {
        //                 auto albedo = random_vec(rand_states, 0.5, 1);
        //                 auto fuzz = random_double(rand_states, 0, 0.5);

        //                 material = new Metal(albedo, fuzz);
        //                 obj_lst[index] = new Sphere(center, 0.2, material);
        //                 index += 1;
                        
        //             } else {
        //                 material = new Dielectric(1.5);
        //                 obj_lst[index] = new Sphere(center, 0.2, material);
        //                 index += 1;
        //             }
        //         }

        //         if (index >= num_spheres - 1) {
        //             break;
        //         }
        //     }
        //     if (index >= num_spheres - 1) {
        //             break;
        //         }

        // }


        // world = new HittableList(obj_lst, index);



        ////////////// Triangle Example //////////////

        // int index = 1;

        // int num_obj = 2;

        // obj_lst = (Hittable**)malloc(num_obj * sizeof(Hittable*));
        
        // obj_lst[0] = new Sphere(Point(0,-1000,0), 1000, new Lambertian(Color(0.5,0.5,0.5)));

        // Material* tri = new Lambertian(Color(0.9, 0.1, 0.1));
        // Point v0(-0.8,  -0.4, -1.5);
        // Point v1( 0.8,  -0.4, -1.5);
        // Point v2( 0.0,   0.8, -1.5);
        // obj_lst[index++] = new Triangle(v0, v1, v2, tri);

        // world = new HittableList(obj_lst, index);

        ////////////// Triangle Example //////////////

        // Reserve: ground + all triangles
        const int num_max = 1 + fcount;
        obj_lst = (Hittable**)malloc(num_max * sizeof(Hittable*));

        // Ground
        obj_lst[0] = new Sphere(Point(0,-1000,0), 1000, new Lambertian(Color(0.5,0.5,0.5)));
        int index = 1;

        // Place/rotate params
        const double rad = yaw_deg * (pi/180.0);
        const double c = cos(rad), s = sin(rad);

        Material* tea_mat = new Lambertian(Color(0.75, 0.72, 0.68)); // clay-ish

        // Build triangles
        for (int f = 0; f < fcount && index < num_max; ++f) {
            int3 tri = faces[f];
            if (tri.x < 0 || tri.y < 0 || tri.z < 0 ||
                tri.x >= vcount || tri.y >= vcount || tri.z >= vcount) continue;

            Point a = vtxf(verts, tri.x);
            Point b = vtxf(verts, tri.y);
            Point c0 = vtxf(verts, tri.z);

            // scale
            a = Point(a.x()*scale,  a.y()*scale,  a.z()*scale);
            b = Point(b.x()*scale,  b.y()*scale,  b.z()*scale);
            c0 = Point(c0.x()*scale, c0.y()*scale, c0.z()*scale);

            // yaw
            a = yaw_y(a, c, s);
            b = yaw_y(b, c, s);
            c0 = yaw_y(c0,c, s);

            // translate
            a = a + translate;
            b = b + translate;
            c0 = c0+ translate;

            obj_lst[index++] = new Triangle(a, b, c0, tea_mat);
        }

        world = new HittableList(obj_lst, index);
    }

}

__global__ void free_world()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        free(obj_lst);
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
            Ray ray = camera.get_ray(col, row, rand_states);
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

    cudaDeviceSetLimit(cudaLimitStackSize, 131070);

    size_t heapBytes = 256 * 1024 * 1024; // 256 MB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapBytes);

    int samples_per_pixel = 50; 

    auto R = cos(pi/4);

    Camera camera;
    
    // camera.aspect_ratio = 16.0 / 9.0;
    // camera.img_width = 1200;
    // camera.vfov = 23;  
    // camera.lookfrom = Point(13,3,5);
    // camera.lookat   = Point(0,0,0);
    // camera.vup      = Vector(0,1,0);
    // camera.defocus_angle = 0.1;
    // camera.focus_dist    = 3.0;

    ///////////// for teapot ///////////////
    camera.aspect_ratio = 16.0 / 9.0;
    camera.img_width = 1200;
    camera.vfov = 15.0;  
    camera.lookfrom = Point(8.0, 2.5, 3.0);
    camera.lookat   = Point(4.0, 1.25, 0.0);
    camera.vup      = Vector(0.0, 1.0, 0.0);
    camera.defocus_angle = 0.0;

    const double dx = camera.lookfrom.x() - camera.lookat.x();
    const double dy = camera.lookfrom.y() - camera.lookat.y();
    const double dz = camera.lookfrom.z() - camera.lookat.z();
    camera.focus_dist    = std::sqrt(dx*dx + dy*dy + dz*dz);
    ///////////// for teapot ///////////////

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

    curandState* rand_states;
    int N = 2048;
    int num_threads = 512;
    int num_blocks = (N + num_threads - 1) / num_threads;
    cudaMalloc((void**)&rand_states, N * sizeof(curandState));

    setup_kernel<<<num_blocks, num_threads>>>(rand_states, time(0));

    /////////////// teapot ///////////////
    // --- Load OBJ on host ---
    ObjLoader teapot;
    if (!teapot.load("/mnt/workspace/obj/teapot.obj")) {  // <- adjust path
        std::cerr << "Failed to load teapot.obj\n";
        return 1;
    }
    teapot.normalize();

    // Upload to device (Unified Memory)
    float* d_vertices = nullptr;
    int3*  d_faces    = nullptr;
    int    vcount = 0, fcount = 0;
    teapot.upload_to_device(d_vertices, vcount, d_faces, fcount);

    // Optional placement: size & pose in your scene
    double S = 2.5;                      // try 2–3
    Point  T(0.0, 0.4, -3.0);            // lift + push forward
    double yaw_deg = 0.0;                // try 90.0 if facing sideways

    // --- Build world with teapot ---
    create_world<<<1,1>>>(rand_states, d_vertices, vcount, d_faces, fcount, S, T, yaw_deg);
    
    GPU_ERR_CHECK(cudaGetLastError());
    GPU_ERR_CHECK(cudaDeviceSynchronize());
    /////////////// teapot ///////////////

    // create_world<<<1,1>>>(rand_states);

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
    cudaFree(d_vertices);
    cudaFree(d_faces);
    delete[] h_img.data;

    return 0;
}
