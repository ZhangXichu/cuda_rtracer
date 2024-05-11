# Use an official NVIDIA CUDA image as the base image
# This image includes CUDA and cuDNN libraries
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install needed packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk-3-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libeigen3-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    sphinx-common \
    yasm \
    libfaac-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenexr-dev \
    libgstreamer-plugins-base1.0-dev \
    libavutil-dev \
    libavfilter-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory 
WORKDIR /mnt/workspace

# Set environment variables
# To make sure the binaries and libraries of CUDA toolkit are in PATH
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Define default command to keep the container running
CMD ["bash"]