#define TEST_NAME "base/image/image_test"
#include "utils/unit_test.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "base/image/Image.h"

BOOST_AUTO_TEST_CASE(TestImageConsistencyWithOpenCVMat) {
  cv::Mat cv_img(640, 480, CV_8UC3);
  cv::RNG rng;
  rng.fill(cv_img, cv::RNG::UNIFORM, 0, 100);

  std::cout << 1 << std::endl;
  my3d::base::Image my3d_img(640, 480, 3, cv_img.data);

  for (size_t row = 0UL; row < 640; ++row) {
    for (size_t col = 0UL; col < 480; ++col) {
      for (size_t channel = 0; channel < 3; ++channel) {
        // std::cout << "Comparing element at (" << row << ", " << col << ", "
        // << channel <<") ..."; std::cout << my3d_img.at(row, col, channel) <<
        // ", "; std::cout << cv_img.ptr<cv::Vec3b>(row, col)[channel] <<
        // std::endl;

        BOOST_CHECK_EQUAL(cv_img.at<cv::Vec3b>(row, col)[channel],
                          my3d_img.at(row, col, channel));
      }
    }
  }
}
