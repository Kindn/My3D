#include "Estimator.h"
#include "utils/TicToc.h"
#include "visualization.h"

#include <opencv4/opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <tuple>

using namespace my3d;
using namespace my3d::slam::simple_vo;

int main(int argc, char **argv) {
  std::string const video_path{argv[1]};
  std::cout << "[INFO] video path: " << video_path << std::endl;
  int32_t const wait_time{std::atoi(argv[2])};

  cv::VideoCapture cap{video_path};
  if (!cap.isOpened()) {
    std::cout << "[ERROR] Failed to open video " << video_path << std::endl;
    return -1;
  }

  double const fps{cap.get(cv::CAP_PROP_FPS)};
  double const dt{1.0 / fps};
  std::cout << "[INFO] FPS: " << fps << ", dt: " << dt << std::endl;
  double const frame_width{cap.get(cv::CAP_PROP_FRAME_WIDTH)};
  double const frame_height{cap.get(cv::CAP_PROP_FRAME_HEIGHT)};
  std::cout << "[INFO] frame size (rxc): " << int(frame_height) << "x"
            << int(frame_width) << std::endl;

  Estimator::Config estimator_config{};
  Eigen::Matrix3d K{};
  K << 600.0, 0.0, frame_width / 2.0, 0.0, 600.0, frame_height / 2.0, 0.0, 0.0,
      1.0;
  estimator_config.feature_tracker_config.camera =
      std::make_shared<base::RadialPinHoleCamera>(K, 0.0, 0.0);
  estimator_config.init_e_ransac_config.max_inlier_error =
      estimator_config.feature_tracker_config.camera->thresholdPix2Norm(4.0);
  estimator_config.init_e_ransac_config.min_iter_num = 30UL;
  estimator_config.init_e_ransac_config.min_inlier_ratio = 0.25;
  estimator_config.init_e_ransac_config.confidence = 0.999;
  estimator_config.init_h_ransac_config.max_inlier_error =
      estimator_config.feature_tracker_config.camera->thresholdPix2Norm(4.0);
  estimator_config.init_h_ransac_config.min_iter_num = 30UL;
  estimator_config.init_h_ransac_config.min_inlier_ratio = 0.25;
  estimator_config.init_h_ransac_config.confidence = 0.999;
  estimator_config.min_avg_parallax_init_deg = 0.8;
  auto const estimator{std::make_shared<Estimator>(estimator_config)};
  std::shared_ptr<EstimatorVisualizer> visualizer{
      std::make_shared<EstimatorVisualizer>(estimator, "Simple VO")};
  visualizer->run();

  cv::Mat frame_cv{};
  double timestamp{0.0};
  util::TicToc tic_toc{};
  while (cap.isOpened()) {
    cap >> frame_cv;
    // cv::imshow("frame raw", frame_cv);
    cv::Mat frame_gray_cv{};
    if (frame_cv.channels() == 3) {
      cv::cvtColor(frame_cv, frame_gray_cv, cv::COLOR_BGR2GRAY);
    }
    std::shared_ptr<base::Image> const frame{std::make_shared<base::Image>(
        frame_cv.rows, frame_cv.cols, frame_gray_cv.channels(),
        frame_gray_cv.data)};
    std::cout << "[INFO] Processing frame... "
              << "timestamp: " << timestamp << "----------------------------"
              << std::endl;
    tic_toc.tic();
    estimator->feedFrame(frame, timestamp);
    std::cout << "[INFO] Processing done. time: " << tic_toc.toc() << "sec(s)"
              << std::endl;
    Eigen::Quaterniond rotation{};
    Eigen::Vector3d translation{};
    if (estimator->getTrackedPose(rotation, translation)) {
      std::cout << "[INFO] "
                << "pose: [" << translation.transpose() << "], "
                << "[" << rotation.coeffs().transpose() << "]" << std::endl;
    } else {
      std::cout << "[ERROR] Tracking failed. " << std::endl;
      cv::waitKey(0);
    }
    timestamp += dt;
    visualizer->update();

    estimator->drawTrackingResult(frame_cv);
    cv::imshow("Tracking Result", frame_cv);

    cv::waitKey(wait_time);
  }

  visualizer->join();

  return 0;
}
