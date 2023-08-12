ARG BASE_IMAGE=ubuntu:22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive


# Install any required dependencies and libraries using package manager (apt, yum, etc.)
RUN apt-get update && \
    apt-get install -y libopencv-dev cmake gcc-12 g++12 build-essential nvidia-driver-$(cat /sys/module/nvidia/version | grep -o -P '\d{3}') nvidia-utils-$(cat /sys/module/nvidia/version | grep -o -P '\d{3}')


# Set the compiler environment variable
ENV CXX=/usr/bin/g++ \
    PATH=/usr/local/nvidia/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 :${LD_LIBRARY_PATH}

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility \
nvidia-smi

# Copy your current directory contents into the container at /app
COPY . .

RUN if [ ! -d build ]; then mkdir build; fi

RUN cd build && cmake .. && make


