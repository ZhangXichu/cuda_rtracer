# Use an official NVIDIA CUDA image as the base image
# This image includes CUDA and cuDNN libraries
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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

# Download OpenCV and OpenCV_contrib
RUN git clone https://github.com/opencv/opencv.git /opt/opencv && \
    git clone https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib

# Checkout the latest version or specific tag
WORKDIR /opt/opencv
RUN git checkout master

WORKDIR /opt/opencv_contrib
RUN git checkout master

# Create a build directory
WORKDIR /opt/opencv
RUN mkdir build

# Configure the build with CMake
WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN="6.1" \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..

# Compile and install
RUN make -j$(nproc)
RUN make install

# Set the working directory back to the root
WORKDIR /

# Set environment variables
# To make sure the binaries and libraries of CUDA toolkit are in PATH
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Define default command to keep the container running
CMD ["bash"]