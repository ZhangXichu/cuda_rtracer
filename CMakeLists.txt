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
                src/matrix.cu
                src/sphere.cu
                src/hittable_list.cu
                src/interval.cu
                src/camera.cu
                src/material.cu
                src/triangle.cu
                src/obj_loader.cu)

set_target_properties(rtracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(rtracer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:DEBUG>:-g>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:DEBUG>:-G -g -lineinfo>>
    $<$<CONFIG:RELEASE>:-O3>
)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

target_link_libraries(rtracer PRIVATE
    $<$<CONFIG:DEBUG>:>
    $<$<CONFIG:RELEASE>:>
)