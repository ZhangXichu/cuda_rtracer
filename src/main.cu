#include <iostream>
#include <fstream>
#include <cuda_runtime.h>


__global__ void write_img(uchar3* d_img, int height, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
    {
        double r = double(col) / (width - 1);
        double g = double(row) / (height - 1);
        double b = 0.0f;

        d_img[row * width + col].x = static_cast<unsigned char>(255.999 * r);
        d_img[row * width + col].y = static_cast<unsigned char>(255.999 * g);
        d_img[row * width + col].z = static_cast<unsigned char>(255.999 * b);
    }
}

int main()
{
    int img_width = 256;
    int img_height = 256;

    const int block_size = 16;

    cudaError_t error;

    uchar3* d_img;
    size_t size = img_width * img_height * sizeof(uchar3);

    uchar3* h_img;
    h_img = new uchar3[size];


    error = cudaMalloc(&d_img, size);
    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    dim3 dim_block(block_size, block_size);

    dim3 dim_grid((img_width + dim_block.x-1) / block_size , (img_height + dim_block.y-1) / block_size ); 

    write_img<<<dim_grid, dim_block>>>(d_img, img_height, img_width);
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    error = cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error during cudaMemcpy: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::ofstream ofs("output.ppm", std::ios::out | std::ios::binary);
    ofs << "P3\n" << img_width << ' ' << img_height << "\n255\n";
    for (int j = 0; j < img_height; j++) {
        for (int i = 0; i < img_width; i++) {
            int index = j * img_width + i;
            ofs << (int)h_img[index].x << ' '
                      << (int)h_img[index].y << ' '
                      << (int)h_img[index].z << '\n';
        }
    }

    cudaFree(d_img);
    delete[] h_img;

    return 0;
}