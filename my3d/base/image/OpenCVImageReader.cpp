/*
 * filename: OpenCVImageReader.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "base/image/OpenCVImageReader.h"

namespace my3d {
namespace base {

bool OpenCVImageReader::read(const std::string &path, Image &out,
                             ImReadMode mode) const {
  std::cout << "[OpenCVImageReader] Reading image from " << path
            << " --------------" << std::endl;

  cv::Mat cv_img;
  if (mode == ImReadMode::IMREAD_COLOR) {
    cv_img = cv::imread(path, cv::IMREAD_COLOR);
  } else if (mode == ImReadMode::IMREAD_GRAYSCALE) {
    cv_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
  }
  out.clear();

  if (cv_img.empty()) {
    return false;
  } else {
    std::ifstream ifs;
    ifs.open(path, std::ios::binary);
    if (!ifs.is_open()) {
      return false;
    } else {
      TinyEXIF::EXIFInfo &exif_info = out.getEXIFInfo();
      exif_info.parseFrom(ifs);
      if (exif_info.Fields) {
        std::cout << "EXIF Information: \n"
                  << "\tImage Description: " << exif_info.ImageDescription
                  << "\n"
                  << "\tImage Resolution: " << cv_img.cols << "x" << cv_img.rows
                  << " pixels\n"
                  << "\tCamera Model: " << exif_info.Make << " - "
                  << exif_info.Model << "\n"
                  << "\tFocal Length: " << exif_info.FocalLength << " mm\n"
                  << "\tFocal Length in 35mm: "
                  << exif_info.LensInfo.FocalLengthIn35mm << " mm" << std::endl;
      }

      double fx_prior = fx_prior_, fy_prior = fy_prior_;
      double cx_prior = cv_img.cols / 2.0, cy_prior = cv_img.rows / 2.0;
      if (fx_prior < 0.0) {
        fx_prior = 1.5 * cv_img.cols;
      }
      if (fy_prior < 0.0) {
        fy_prior = 1.5 * cv_img.rows;
      }

      int intrinsic_prior_src =
          extractIntrinsicsFromEXIF(exif_info, fx_prior, fy_prior, cx_prior,
                                    cy_prior, cv_img.cols, cv_img.rows);
      switch (intrinsic_prior_src) {
      case FocalLengthPriorType::FROM_CAMERA_DATABASE:
        std::cout << "Obtained intrinsic prior from camera database: "
                  << std::endl;
        break;
      case FocalLengthPriorType::FROM_EXIF_CALIB_INFO:
        std::cout
            << "Obtained intrinsic prior from EXIF calibration information: "
            << std::endl;
        break;
      case FocalLengthPriorType::FROM_EXIF_FOCAL_LENGTH_35MM:
        std::cout << "Obtained intrinsic prior from EXIF focal length in 35mm: "
                  << std::endl;
        break;
      case FocalLengthPriorType::FROM_DEFAULT_VALUE:
        std::cout << "Obtained intrinsic prior from default values: "
                  << std::endl;
        break;
      default:
        break;
      }
      std::cout << "\tfx: " << fx_prior << " pixels\n"
                << "\tfy: " << fy_prior << " pixels\n"
                << "\tcx: " << cx_prior << " pixels\n"
                << "\tcy: " << cy_prior << " pixels" << std::endl;

      Eigen::Matrix3d camera_matrix_prior;
      camera_matrix_prior << fx_prior, 0, cx_prior, 0, fy_prior, cy_prior, 0, 0,
          1;
      CameraBase *camera = new RadialPinHoleCamera(camera_matrix_prior);

      out.set(cv_img.rows, cv_img.cols, cv_img.channels(), cv_img.data, 0, path,
              camera);
    }
  }

  return true;
}

bool OpenCVImageReader::save(const std::string &path, const Image &src) const {
  if (src.isEmpty()) {
    return false;
  }

  int type;
  if (src.channels() == 1) {
    type = CV_8UC1;
  } else if (src.channels() == 2) {
    type = CV_8UC2;
  } else if (src.channels() == 3) {
    type = CV_8UC3;
  } else if (src.channels() == 4) {
    type = CV_8UC4;
  }

  cv::Mat cv_img(static_cast<int>(src.rows()), static_cast<int>(src.cols()),
                 type);
  std::copy(src.data(), src.data() + src.size() * src.channels(), cv_img.data);
  return cv::imwrite(path, cv_img);
}

} // namespace base
} // namespace my3d