cmake_minimum_required(VERSION 3.18)

project(HighPerfCV VERSION 1.0 LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations (x86 only)" ON)
option(ENABLE_NEON "Enable NEON optimizations (ARM only)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_PROCESSOR)

if(SYSTEM_PROCESSOR MATCHES "amd64|x86_64|i386|i686")
    set(IS_X86 TRUE)
    message(STATUS "Detected Architecture: x86/x64 (${CMAKE_SYSTEM_PROCESSOR})")
elseif(SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(IS_ARM TRUE)
    message(STATUS "Detected Architecture: ARM (${CMAKE_SYSTEM_PROCESSOR})")
else()
    message(STATUS "Detected Architecture: Other (${CMAKE_SYSTEM_PROCESSOR})")
endif()

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 53 61 62 72 75 86 87 89)
        endif()

        find_package(CUDAToolkit REQUIRED)
        add_compile_definitions(USE_CUDA)
        message(STATUS "CUDA detected and enabled.")
    else()
        message(STATUS "No NVCC compiler found. Disabling CUDA support.")
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
        src/filters/grayscale/grayscale_omp.cpp
        src/filters/grayscale/grayscale_omp.h

        src/filters/brightness/brightness_base.cpp
        src/filters/brightness/brightness_base.h
        src/filters/brightness/brightness_omp.cpp
        src/filters/brightness/brightness_omp.h

        src/filters/gaussian_blur/gaussian_base.cpp
        src/filters/gaussian_blur/gaussian_base.h
        src/filters/gaussian_blur/gaussian_omp.cpp
        src/filters/gaussian_blur/gaussian_omp.h

        src/filters/median/median_base.cpp
        src/filters/median/median_base.h

        src/filters/median/median_omp.cpp
        src/filters/median/median_omp.h
)

if(ENABLE_CUDA)
    list(APPEND SRCS
            src/filters/grayscale/grayscale_kernel.cu
            src/filters/grayscale/grayscale_kernel.h
            src/filters/grayscale/grayscale_cuda.cpp
            src/filters/grayscale/grayscale_cuda.h
            src/filters/brightness/brightness_cuda.h
            src/filters/brightness/brightness_cuda.cpp
            src/filters/brightness/brightness_kernel.cu
            src/filters/brightness/brightness_kernel.h
            src/filters/gaussian_blur/gaussian_cuda.h
            src/filters/gaussian_blur/gaussian_cuda.cpp
            src/filters/gaussian_blur/gaussian_kernel.cu
            src/filters/gaussian_blur/gaussian_kernel.h
    )
endif()

add_executable(HighPerfCV ${SRCS})

target_include_directories(HighPerfCV PRIVATE ${OpenCV_INCLUDE_DIRS})

include(CheckCXXCompilerFlag)

if(IS_X86 AND ENABLE_AVX2)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" HAS_AVX2_FLAG)
        if(HAS_AVX2_FLAG)
            target_compile_options(HighPerfCV PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
            )

            target_compile_definitions(HighPerfCV PRIVATE USE_AVX2)

            target_sources(HighPerfCV PRIVATE
                    src/filters/grayscale/grayscale_avx.h
                    src/filters/grayscale/grayscale_avx.cpp
                    src/filters/brightness/brightness_avx.h
                    src/filters/brightness/brightness_avx.cpp
                    src/filters/gaussian_blur/gaussian_avx.h
                    src/filters/gaussian_blur/gaussian_avx.cpp
            )
            message(STATUS "AVX2 optimizations enabled (MSVC).")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" HAS_AVX2_FLAG)
        if(HAS_AVX2_FLAG)
            target_compile_options(HighPerfCV PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
                    $<$<COMPILE_LANGUAGE:CXX>:-mfma>
            )
            target_compile_definitions(HighPerfCV PRIVATE USE_AVX2)

            target_sources(HighPerfCV PRIVATE
                    src/filters/grayscale/grayscale_avx.h
                    src/filters/grayscale/grayscale_avx.cpp
                    src/filters/brightness/brightness_avx.h
                    src/filters/brightness/brightness_avx.cpp
                    src/filters/gaussian_blur/gaussian_avx.h
                    src/filters/gaussian_blur/gaussian_avx.cpp
            )
            message(STATUS "AVX2 optimizations enabled (GCC/Clang).")
        endif()
    endif()
endif()

if(IS_ARM AND ENABLE_NEON)
    if(NOT MSVC)
        target_compile_options(HighPerfCV PRIVATE -march=native -O3)
        target_compile_definitions(HighPerfCV PRIVATE USE_NEON)
        message(STATUS "ARM NEON optimizations enabled (flags imply native arch).")
    endif()
endif()

if(ENABLE_CUDA)
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