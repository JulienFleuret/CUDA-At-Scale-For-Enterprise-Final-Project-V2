cmake_minimum_required(VERSION 3.8)

project(cudaAtScaleV2 LANGUAGES CXX)

# First set cuda.
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(CUDA REQUIRED)

# Set a few variable
set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_SDK_TOOLKIT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})



# Automatic finding of the architecture"

# Find nvidia-smi executable
find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)

# Check if nvidia-smi was found
if(NOT NVIDIA_SMI_EXECUTABLE)
    message(FATAL_ERROR "nvidia-smi not found!")
endif()

# Step 1) Execute nvidia-smi command to get the architectures
execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv OUTPUT_VARIABLE GPU_INFO)

# Step 2) Extract GPU architectures from the output
string(REGEX MATCHALL "[0-9]+\\.[0-9]+" GPU_ARCHITECTURES "${GPU_INFO}")

# Step 3) Remove the dot in the middle of the architectures
foreach(ARCHITECTURE IN LISTS GPU_ARCHITECTURES)
    string(REPLACE "." "" ARCHITECTURE_WITHOUT_DOT ${ARCHITECTURE})
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${ARCHITECTURE_WITHOUT_DOT})
endforeach()

# Then Enable CUDA.
enable_language(CUDA)


#Find the proper OpenCV

#Check if a local instalation exists or not and adjust the path to the cmake configuration file accordingly.
if (EXISTS /usr/local/opencv)
set(OpenCV_CMAKE_ROOT_DIR /usr/local/opencv/lib/cmake)
elseif(EXISTS /usr/lib/x86_64-linux-gnu/cmake)
set(OpenCV_CMAKE_ROOT_DIR /usr/lib/x86_64-linux-gnu/cmake)
endif()

#Find where is the file. This step is disociate from the previous because the folder after OpenCV_CMAKE_ROOT_DIR has a version in its naming.
execute_process(
    COMMAND find ${OpenCV_CMAKE_ROOT_DIR} -name OpenCVConfig.cmake -type f -print
    OUTPUT_VARIABLE OpenCVConfigFile
    OUTPUT_STRIP_TRAILING_WHITESPACE)


#Keep only the path.
get_filename_component(OpenCV_DIR ${OpenCVConfigFile} DIRECTORY)

#Look for OpenCV.
find_package(OpenCV REQUIRED core imgcodecs highgui)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-D__WARMUP_CALL__)

add_executable(cudaAtScaleV2 src/main.cu src/types.cuh src/types.cu src/utils.cuh src/utils.cu)

set_target_properties(cudaAtScaleV2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_EXECUTABLE_SUFFIX ".exe")


set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_FLAGS "-m64 --compiler-options -fno-strict-aliasing -use_fast_math --std=c++17 --expt-relaxed-constexpr --extended-lambda -O3 --display-error-number --source-in-ptx")

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/bin)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endif()

set_target_properties(cudaAtScaleV2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


target_include_directories(cudaAtScaleV2 PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_directories(cudaAtScaleV2 PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib)
target_link_libraries(cudaAtScaleV2 PRIVATE -lcudart -lnppc -lnppif -lnppisu)

target_include_directories(cudaAtScaleV2 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_directories(cudaAtScaleV2 PRIVATE ${OpenCV_INCLUDE_DIRS}/../../lib)
target_link_libraries(cudaAtScaleV2 PRIVATE -lopencv_core -lopencv_imgcodecs -lopencv_highgui)


install(TARGETS cudaAtScaleV2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
