# HighPerfCV: High-Performance Computer Vision Framework

**HighPerfCV** is a cross-platform C++20 image processing framework designed to demonstrate the impact of hardware-aware optimizations. It implements classic computer vision algorithms using four distinct optimization levels: **Scalar (Base)**, **Multi-threading (OpenMP)**, **Vectorization (AVX2 / NEON)**, and **GPU Acceleration (CUDA)**.

The project highlights the transition from $O(N^2)$ naive implementations to highly optimized, memory-aware, and instruction-level parallel solutions suitable for both **x86 Desktop** and **ARM Embedded (NVIDIA Jetson)** platforms.

---

## 🚀 Key Features

* **Cross-Architecture Support:**
    * **x86_64:** Utilizes **AVX2** intrinsics (FMA, bitwise sorting networks).
    * **ARM64 (Jetson):** Utilizes **NEON** intrinsics (structural loading, fixed-point arithmetic).
* **CUDA Acceleration:** Custom kernels using Shared Memory, Constant Memory, and Warp-level optimizations.
* **Modern C++ Design:** Object-Oriented architecture with `IFilter` interface, Factory Pattern, and RAII memory management.
* **Benchmarking Ready:** Built-in performance measurement and hardware capability detection.

---

## 📊 Performance Benchmarks

*Hardware: Intel Core i5-14400F (Host) | NVIDIA GTX 1060 6GB (GPU) | Image: 24 MPx (6000x4000)*

| Algorithm | Mode | Time (ms) | Speedup | Key Optimization Technique |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian Blur** | Base | ~34,000 | 1x | Naive 2D Convolution $O(K^2)$ |
| | OpenMP | ~1,200 | ~28x | Separable Filter 1D+1D + Multi-threading |
| | **AVX2** | **555** | **~61x** | **FMA Instructions + Vectorized Vertical Pass** |
| | **CUDA** | **120** | **~283x** | **`__constant__` Memory for Kernel + Massive Parallelism** |
| | | | | |
| **Median Filter** | Base | ~45,000 | 1x | `std::sort` per pixel (Branch heavy) |
| | OpenMP | ~6,000 | ~7x | Thread-local buffers to avoid heap contention |
| | **AVX2** | **154** | **~290x** | **Branchless 3x3 Sorting Network (Min/Max instructions)** |
| | **CUDA** | **TBD** | **High** | **Register-based Sorting Network** |

---

## 🛠️ Tech Stack & Implementation Details

### 1. Algorithms Implemented

#### 🌑 Grayscale Conversion
* **Base:** Floating point multiplication.
* **NEON (ARM):** Uses `vld3` for structural de-interleaving of RGB channels and fixed-point arithmetic to avoid floating-point overhead.
* **AVX2 (x86):** Vectorized weighted sum.

#### ☀️ Brightness & Contrast
* **Base:** Simple linear transform $g(x) = \alpha f(x) + \beta$ with `saturate_cast`.
* **CUDA:** Memory-bound operation optimized using grid-stride loops to maximize memory bandwidth saturation.

#### 🌫️ Gaussian Blur
* **Algorithmic Optimization:** Converted from 2D convolution ($K^2$ ops/pixel) to Separable 1D convolution ($2K$ ops/pixel).
* **SIMD:** The vertical pass uses AVX2 FMA (`_mm256_fmadd_ps`) to process 8 pixels per cycle.
* **CUDA:** Kernel weights stored in Constant Memory to reduce global memory reads; coalesced memory access patterns.

#### ⚡ Median Filter (3x3)
* **Challenge:** Sorting is branch-heavy and causes pipeline flushes on CPUs.
* **Optimization:** Implemented a **Sorting Network** using only `min` and `max` instructions. This makes the code **branchless**, allowing the CPU pipeline to execute at full throughput without mispredictions.

---

## ⚙️ Build Instructions

### Prerequisites
* **C++20 Compiler** (GCC, Clang, or MSVC)
* **CMake 3.18+**
* **OpenCV 4.x** (Used for I/O and display only)
* **CUDA Toolkit** (Optional, for GPU support)

### Building (Linux / Jetson / Windows)

```bash
mkdir build && cd build
cmake .. 
# CMake automatically detects CPU arch (x86/ARM) and CUDA availability.
make -j
```

---

## 🖥️ Usage

# Syntax
```bash
./HighPerfCV <image_path> <FILTER_TYPE> <MODE> [params...]
```

# Examples
```bash
./HighPerfCV input.jpg GAUSSIAN_BLUR CUDA 5 1.0
```
```bash
./HighPerfCV photo.png GRAYSCALE NEON
```
```bash
./HighPerfCV frame.jpg MEDIAN AVX2 3
```
```bash
./HighPerfCV dark.jpg BRIGHTNESS OPENMP 1.5 20
```
