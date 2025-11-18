#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#ifdef USE_CUDA
    #include <cuda_runtime.h>
#endif

#include "../include/IFilter.h"
#include "../include/utils.h"
#include "../include/FilterFactory.h"

void display_results(const cv::Mat& inputImage, const cv::Mat& outputImage, const std::string& filter_name, long long duration_ms) {
    cv::Mat display_output;

    if (inputImage.channels() == 3 && outputImage.channels() == 1) {
        cv::cvtColor(outputImage, display_output, cv::COLOR_GRAY2BGR);
    } else {
        display_output = outputImage;
    }

    cv::Mat comparison_image;
    cv::hconcat(inputImage, display_output, comparison_image);

    const std::string text = filter_name + " | Time: " + std::to_string(duration_ms) + " ms";
    cv::putText(comparison_image, text, cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Comparison: Input vs. Output", comparison_image);
    std::cout << "Press any key to exit..." << '\n';
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Error: Not enough arguments.\n";
        std::cerr << "Usage: " << argv[0] << " <input_image_path> <filter_type> <optimization_mode>\n";
        return -1;
    }

    bool has_avx2 = cv::checkHardwareSupport(CV_CPU_AVX2);

    int cuda_devices = 0;
    bool cuda_runtime_ok = false;

#ifdef USE_CUDA
    cudaError_t status = cudaGetDeviceCount(&cuda_devices);
    if (status == cudaSuccess && cuda_devices > 0) {
        cuda_runtime_ok = true;
    } else {
        cuda_devices = 0;
    }
#endif

    std::cout << "--- Hardware Detection ---" << std::endl;
    std::cout << "AVX2 support: " << (has_avx2 ? "YES" : "NO") << std::endl;

#ifdef USE_CUDA
    std::cout << "Available CUDA devices: " << cuda_devices << std::endl;
#else
    std::cout << "CUDA support: DISABLED at compile time" << std::endl;
#endif
    std::cout << "----------------------------" << std::endl;

    try {
        const std::string image_path = argv[1];
        const FilterType filter_type = string_to_filter(argv[2]);
        const OptimizationMode opt_mode = string_to_mode(argv[3]);

        if (opt_mode == OptimizationMode::AVX2 && !has_avx2) {
            throw std::runtime_error("FATAL ERROR: AVX2 mode requested, but hardware does not support it.");
        }

        if (opt_mode == OptimizationMode::CUDA) {
            #ifndef USE_CUDA
                throw std::runtime_error("FATAL ERROR: CUDA mode requested, but application was compiled without CUDA support.");
            #else
                if (!cuda_runtime_ok) {
                    throw std::runtime_error("FATAL ERROR: CUDA mode requested, but no compatible NVIDIA device found (or driver error).");
                }
            #endif
        }

        const cv::Mat input_img = cv::imread(image_path);
        if (input_img.empty()) {
            throw std::runtime_error("Could not load image from path: " + image_path);
        }
        std::cout << "Image loaded: " << image_path << " (" << input_img.cols << "x" << input_img.rows << ")" << '\n';

        std::cout << "Creating a filter...\n";
        const std::unique_ptr<IFilter> filter = FilterFactory::create_filter(filter_type, opt_mode);

        if (!filter) {
             throw std::runtime_error("Filter Factory returned null. Unknown filter type or mode?");
        }

        const std::string filter_name = filter->get_name();
        std::cout << "Running the filter: " << filter_name << '\n';

        cv::Mat output_img;

        const auto start = std::chrono::high_resolution_clock::now();
        filter->process(input_img, output_img);
        const auto stop = std::chrono::high_resolution_clock::now();

        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " " << duration.count() << " ms\n";

        display_results(input_img, output_img, filter_name, duration.count());

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return -1;
    }

    return 0;
}