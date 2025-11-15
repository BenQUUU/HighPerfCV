cmake_minimum_required(VERSION 3.18)
project(HighPerfCV VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv)

find_package(OpenCV REQUIRED COMPONENTS
        core highgui imgcodecs imgproc
        NO_DEFAULT_PATH
)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(HighPerfCV
        main.cpp
        kernel.cu
        kernel.cuh
)

set_target_properties(HighPerfCV PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_PROPAGATE_HOST_FLAGS OFF
)

target_include_directories(HighPerfCV PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(HighPerfCV PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2>
        >
)

target_link_libraries(HighPerfCV PRIVATE
        CUDA::cudart
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
)

set(OPENCV_BIN_DIR ${OpenCV_DIR}/x64/vc16/bin)

file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/*.dll")

foreach(DLL_FILE ${OPENCV_DLLS})
    add_custom_command(
            TARGET HighPerfCV POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_FILE}"
            $<TARGET_FILE_DIR:HighPerfCV>
    )
endforeach()