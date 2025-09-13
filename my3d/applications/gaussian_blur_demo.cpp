#include <iostream>

#include "base/image/OpenCVImageReader.h"
#include "base/image/image_processing.h"
#include "utils/TicToc.h"

using namespace my3d; 

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Error! Invalid argument. Usage: \n" << 
                     "\tgaussian_blur_demo <image_directory>" << std::endl; 
        return -1;
    }
    const std::string image_path = argv[1]; 

    base::OpenCVImageReader image_reader; 
    base::Image image; 
    image_reader.read(image_path, image, base::ImageReaderBase::IMREAD_COLOR); 
    std::cout << "image size: " << image.channels() << "x" << image.rows() << "x" << image.cols() << std::endl; 

    base::Image blurred, scaled; 

    util::TicToc tictoc; 

    tictoc.tic();
    base::blurGaussian(image, blurred, 1); 
    std::cout << "Blur Gaussian 1: " << tictoc.toc() << std::endl; 
    image_reader.save("blurred_1.jpg", blurred); 

    tictoc.tic();
    base::blurGaussian(image, blurred, 8); 
    std::cout << "Blur Gaussian 8: " << tictoc.toc() << std::endl; 
    image_reader.save("blurred_8.jpg", blurred);

    cv::Mat const image_cv{cv::imread(image_path)};
    cv::Mat blurred_cv{};
    tictoc.tic();
    cv::GaussianBlur(image_cv, blurred_cv, cv::Size(49, 49), 8.0, 8.0, cv::BORDER_CONSTANT);
    std::cout << "Blur Gaussian CV 8: " << tictoc.toc() << std::endl; 
    cv::imwrite("blurred_8_cv.jpg", blurred_cv);

    // tictoc.tic();
    // base::blurGaussianOld(image, blurred, 8); 
    // std::cout << "Blur Gaussian Old 8: " << tictoc.toc() << std::endl; 
    // image_reader.save("blurred_8_old.jpg", blurred);

    // base::scaleImage(image, scaled, 0.7); 
    // image_reader.save("scaled_0_7.jpg", scaled); 
    // base::scaleImage(image, scaled, 1.7); 
    // image_reader.save("scaled_1_7.jpg", scaled); 

    // base::Image gray; 
    // base::convertImageColorBGR2GrayScale(image, gray); 
    // image_reader.save("gray.jpg", gray); 

    return 0; 
}