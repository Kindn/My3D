#include "Estimator.h"
#include "utils/TicToc.h"
#include "utils/io.h"
#include "visualization.h"

#include <opencv4/opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <tuple>

using namespace my3d;
using namespace my3d::slam::simple_vo;

std::map<std::string, double> readTimestamps(std::string const &data_dir) {
  boost::filesystem::path const path{boost::filesystem::path(data_dir) /
                                     "times.txt"};
  std::map<std::string, double> timestamps{};
  std::ifstream ifs(path.c_str());
  std::string line{};
  while (std::getline(ifs, line)) {
    std::istringstream ss{line};
    std::array<std::string, 3UL> fields{};
    ss >> fields[0] >> fields[1] >> fields[2];
    timestamps.emplace(fields[0], std::stod(fields[1]));
  }

  return timestamps;
}

int main(int argc, char **argv) {
  std::string const data_dir{argv[1]};
  std::cout << "[INFO] data_dir: " << data_dir << std::endl;
  int32_t const wait_time{std::atoi(argv[2])};

  // boost::filesystem::path const image_dir{boost::filesystem::path(data_dir) /
  //                                         "images"};
  boost::filesystem::path const image_dir{boost::filesystem::path(data_dir)};
  std::unordered_set<std::string> const image_exts{".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"};
  std::vector<std::string> image_files{
      util::collectFiles(image_dir.c_str(), image_exts)};
  std::sort(image_files.begin(), image_files.end());
  size_t const num_frames{image_files.size()};
  std::cout << "[INFO] Collected " << num_frames << " images. " << std::endl;
  if (num_frames < 2UL) {
    std::cout << "[ERROR] Too few images. Aborted. " << std::endl;
    return -1;
  }
  // auto const timestamps{readTimestamps(data_dir)};
  // if (timestamps.size() != num_frames) {
  //   std::cout << "[ERROR] Data info mismatched. Aborted. " << std::endl;
  //   return -1;
  // }

  cv::Mat const first_frame{cv::imread(image_files.front(), cv::IMREAD_COLOR)};

  double const frame_width{first_frame.cols};
  double const frame_height{first_frame.rows};
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
  for (size_t i{0UL}; i < num_frames; ++i) {
    std::string const path{image_files[i]};
    frame_cv = cv::imread(path, cv::IMREAD_COLOR);
    timestamp = std::stod(boost::filesystem::path(path).filename().string());
    // cv::imshow("frame raw", frame_cv);
    cv::Mat frame_gray_cv{};
    if (frame_cv.channels() == 3) {
      cv::cvtColor(frame_cv, frame_gray_cv, cv::COLOR_BGR2GRAY);
    }
    std::shared_ptr<base::Image> const frame{std::make_shared<base::Image>(
        frame_cv.rows, frame_cv.cols, frame_gray_cv.channels(),
        frame_gray_cv.data)};
    std::cout << "[INFO] Processing frame " << path << ", "
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
    visualizer->update();

    estimator->drawTrackingResult(frame_cv);
    cv::imshow("Tracking Result", frame_cv);

    cv::waitKey(wait_time);
  }

  visualizer->join();

  return 0;
}
