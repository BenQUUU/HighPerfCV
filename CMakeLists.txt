cmake_minimum_required(VERSION 3.18)

project(HighPerfCV VERSION 1.0 LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA support" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)

        find_package(CUDAToolkit REQUIRED)
        message(STATUS "CUDA detected and enabled.")
    else()
        message(STATUS "No NVCC compiler. The project will be built in CPU-only mode.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

if(WIN32)
    set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv)

    find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc NO_DEFAULT_PATH)
else()
    message(STATUS "Searching for system OpenCV installation...")
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)
endif()

find_package(OpenMP REQUIRED)

set(SRCS
        src/main.cpp
        include/IFilter.h
        include/utils.h
        include/FilterFactory.h
        src/filter_factory.cpp
        src/filters/grayscale/grayscale_base.cpp
        src/filters/grayscale/grayscale_base.h
        src/filters/grayscale/grayscale_omp.h
        src/filters/grayscale/grayscale_omp.cpp
        src/filters/grayscale/grayscale_avx.h
        src/filters/grayscale/grayscale_avx.cpp
        src/filters/brightness/brightness_base.cpp
        src/filters/brightness/brightness_base.h
)

if(ENABLE_CUDA)
    list(APPEND SRCS
            src/filters/grayscale/grayscale_kernel.cu
            src/filters/grayscale/grayscale_kernel.h
            src/filters/grayscale/grayscale_cuda.cpp
            src/filters/grayscale/grayscale_cuda.h
    )
endif()

add_executable(HighPerfCV ${SRCS})

target_include_directories(HighPerfCV PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(HighPerfCV PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2>
        >
)

if(ENABLE_CUDA)
    target_compile_definitions(HighPerfCV PRIVATE USE_CUDA)

    set_target_properties(HighPerfCV PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_PROPAGATE_HOST_FLAGS OFF
    )

    target_link_libraries(HighPerfCV PRIVATE CUDA::cudart)
endif()

target_link_libraries(HighPerfCV PRIVATE
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
)

if(WIN32)
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
endif()