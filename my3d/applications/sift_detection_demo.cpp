#include "base/image/OpenCVImageReader.h"
#include "feature/sift.h" 
#include "utils/TicToc.h"

using namespace my3d; 

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Error! Invalid argument. Usage: \n" << 
                     "\tsparse_reconstruction <image_directory>" << std::endl; 
        return -1;
    }
    const std::string image_path = argv[1]; 

    base::OpenCVImageReader image_reader; 
    base::Image image; 
    if (!image_reader.read(image_path, image, base::ImageReaderBase::IMREAD_COLOR)) {
        std::cout << "Failed to read image. " << std::endl; 
        return -1;
    }

    feature::SIFTKeyPointList key_points; 
    feature::SIFTDescriptorList descriptors; 
    feature::SIFTDetector::Config config; 
    config.octave_idx_base_image = 1;
    config.num_octaves = -1;
    config.min_abs_resp = 0.02; 
    config.edge_score_thresh = 10.0; 
    config.verbose = false; 
    feature::SIFTDetector detector(config); 

    util::TicToc tic_toc{};
    tic_toc.tic();
    detector.setImage(image); 
    detector.detect(key_points); 
    detector.compute(key_points, descriptors); 
    std::cout << "Time: " << tic_toc.toc() << "s. " << std::endl;

    base::Image image_with_features; 
    const uint8_t color[3] = {0, 255, 0};
    feature::drawSIFTKeyPoints(image, image_with_features, 
                               key_points, descriptors, 
                               color, 
                               1.0, 1.0); 
    image_reader.save("./sift.jpg", image_with_features); 

    std::sort(descriptors.begin(), descriptors.end(), 
    [](const feature::SIFTDescriptor &des1, const feature::SIFTDescriptor &des2) {
        return des1.scale > des2.scale; 
    }); 

    std::cout << "Detected " << descriptors.size() << " descriptors. " << std::endl;
    // std::cout << "Descriptors: " << std::endl; 
    // for (const auto &des : descriptors) {
    //     std::cout << "===> " << des.point.transpose() << " ";
    //     des.printHist(); 
    // }

    const auto octaves = detector.getOctaves(); 
    for (size_t i = 0; i < octaves.size(); ++i) {
        const auto &octave = octaves[i]; 
        const auto &dog_space = octave.dog_space; 
        for (size_t l = 0; l < dog_space.getNumValidDoGLevels(); ++l) {
            const Eigen::MatrixXf &dog = dog_space[l + 1]; 
            const double max_dog = dog.maxCoeff(); 
            const double min_dog = dog.minCoeff(); 
            const double max_dog_abs = std::max(std::abs(min_dog), std::abs(max_dog)); 
            if (i == 5) {
                // std::cout << "[" << dog << "]" << std::endl; 
            }
            base::Image dog_img(dog.rows(), dog.cols(), 1); 
            for (size_t r = 0; r < dog_img.rows(); ++r) {
                for (size_t c = 0; c < dog_img.cols(); ++c) {
                    dog_img.at(r, c, 0) = 
                        static_cast<uint8_t>(std::abs(dog(r, c)) / max_dog_abs * 255.0); 
                }
            }
            const std::string dog_path = "dog_oct_" + std::to_string(i) + "_level_" + std::to_string(l) + ".jpg"; 
            image_reader.save(dog_path, dog_img); 
        }
    }

    return 0; 
}
