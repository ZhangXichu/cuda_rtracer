cmake_minimum_required(VERSION 3.20)
project(rtracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "61") 
enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

add_executable(rtracer)

target_include_directories(rtracer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_sources(rtracer 
                PRIVATE 
                src/main.cu
                src/matrix.cu)

set_target_properties(rtracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
