#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

int main()
{
    std::string imagePath = "Lenna.png"; // Zmień na ścieżkę do swojego obrazu
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Nie można wczytać obrazu: " << imagePath << std::endl;
        return -1;
    }

    cv::imshow("Display window", image);
    cv::waitKey(0); // Czeka na naciśnięcie klaw

    return 0;
}