#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <matrix.cuh>


__global__ void write_img(Matrix d_img)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d_img.height && col < d_img.width)
    {
        double r = double(col) / (d_img.height - 1);
        double g = double(row) / (d_img.height - 1);
        double b = 0.0f;

        d_img.at(row, col).x = static_cast<unsigned char>(255.999 * r);
        d_img.at(row, col).y = static_cast<unsigned char>(255.999 * g);
        d_img.at(row, col).z = static_cast<unsigned char>(255.999 * b);
    }
}

int main()
{
    int img_width = 256;
    int img_height = 256;

    const int block_size = 16;

    cudaError_t error;

    size_t size = img_width * img_height * sizeof(uchar3);

    Matrix d_img(img_width, img_height), h_img(img_width, img_height);
    
    h_img.data = new uchar3[size];

    error = cudaMalloc(&d_img.data, size);

    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((img_width + dim_block.x-1) / block_size , (img_height + dim_block.y-1) / block_size ); 

    write_img<<<dim_grid, dim_block>>>(d_img);
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    error = cudaMemcpy(h_img.data, d_img.data, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error during cudaMemcpy: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::ofstream ofs("output.ppm", std::ios::out | std::ios::binary);
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
