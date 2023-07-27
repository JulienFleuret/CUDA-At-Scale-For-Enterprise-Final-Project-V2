cmake_minimum_required(VERSION 3.8)

project(cudaAtScaleV2 LANGUAGES CXX)

set(OpenCV_DIR /usr/local/opencv/lib/cmake)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if (CUDA_FOUND)
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

    # Step 4) Join the list into CMAKE_CUDA_ARCHITECTURES
#    list(JOIN GPU_ARCHITECTURES_WITHOUT_DOT " " CMAKE_CUDA_ARCHITECTURES)

    # Then Enable CUDA.
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA NOT FOUND!")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#cuda_add_executable(cudaAtScaleV2 main.cu)

add_executable(cudaAtScaleV2 main.cu types.cuh types.cu utils.cuh utils.cu)




set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_FLAGS "-m64 --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v --std=c++17 --expt-relaxed-constexpr --extended-lambda -O3 --display-error-number --source-in-ptx -v")



target_include_directories(cudaAtScaleV2 PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_directories(cudaAtScaleV2 PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib)
target_link_libraries(cudaAtScaleV2 PRIVATE -lcuda -lcudart -lnppc -lnppicc -lnppif -lnppim -lnppisu -lnpps -lnppial -lnppidei -lnppig -lnppist -lnppitc)
#target_link_libraries(cudaAtScaleV2 PRIVATE -lnppc -lnppicc -lnppif -lnppim -lnppisu -lnpps -lnppial -lnppidei -lnppig -lnppist -lnppitc)

target_include_directories(cudaAtScaleV2 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_directories(cudaAtScaleV2 PRIVATE ${OpenCV_INCLUDE_DIRS}/../../lib)
#target_link_libraries(cudaAtScaleV2 PRIVATE ${OpenCV_LIBS})
target_link_libraries(cudaAtScaleV2 PRIVATE -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui)


install(TARGETS cudaAtScaleV2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
