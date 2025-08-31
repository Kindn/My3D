/*
 * filename: exif.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "base/image/exif.h"

namespace my3d {
namespace base {

int extractIntrinsicsFromEXIF(const TinyEXIF::EXIFInfo &exif_info, double &fx,
                              double &fy, double cx, double cy,
                              const size_t &width, const size_t &height) {
  const double exif_focal_length = exif_info.FocalLength;
  const std::string exif_camera_maker = exif_info.Make;
  const std::string exif_camera_model = exif_info.Model;
  // const uint32_t exif_image_width = exif_info.ImageWidth;
  // const uint32_t exif_image_height = exif_info.ImageHeight;
  double sensor_width_mm = -1.0, sensor_height_mm = -1.0;

  cx = static_cast<double>(width) / 2.0;
  cy = static_cast<double>(height) / 2.0;

  if (exif_focal_length > 0.0 && !exif_camera_model.empty()) {
    CameraDatabase const *db = CameraDatabase::get();
    CameraInfo const *camera_info =
        db->lookup(exif_camera_maker, exif_camera_model);
    if (camera_info != nullptr) {
      sensor_width_mm = camera_info->sensor_width_mm;
      sensor_height_mm = camera_info->sensor_height_mm;
    }
  }

  if (exif_focal_length > 0.0 && sensor_width_mm > 0.0 &&
      sensor_height_mm > 0.0) {
    fx = exif_focal_length * width / sensor_width_mm;
    fy = exif_focal_length * height / sensor_height_mm;
    fx = 0.5 * (fx + fy);
    fy = fx;

    return FocalLengthPriorType::FROM_CAMERA_DATABASE;
  }

  const double exif_focal_length_35mm = exif_info.LensInfo.FocalLengthIn35mm;
  if (exif_focal_length > 0.0 && exif_focal_length_35mm) {
    double sensor_size = 35.0 * exif_focal_length / exif_focal_length_35mm;
    fx = exif_focal_length * width / sensor_size;
    fy = fx;

    return FocalLengthPriorType::FROM_EXIF_FOCAL_LENGTH_35MM;
  }

  const TinyEXIF::EXIFInfo::Calibration_t exif_calib = exif_info.Calibration;
  if (exif_calib.FocalLength > 0.0 && exif_calib.OpticalCenterX > 0.0 &&
      exif_calib.OpticalCenterY > 0.0) {
    fx = exif_calib.FocalLength;
    fy = fx;
    cx = exif_calib.OpticalCenterX;
    cy = exif_calib.OpticalCenterY;

    return FocalLengthPriorType::FROM_EXIF_CALIB_INFO;
  }

  // Set to default values.

  fx = fy = 1.2 * static_cast<double>(std::max(width, height));

  return FocalLengthPriorType::FROM_DEFAULT_VALUE;
}

} // namespace base
} // namespace my3d