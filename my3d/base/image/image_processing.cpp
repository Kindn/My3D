/*
 * filename: image_processing.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:  Some useful DIP algorithms.
 */

#include "base/image/image_processing.h"

namespace my3d {
namespace base {

void blurGaussian(const Image &src, Image &dst, const double &sigma) {
  assert(sigma > 0);
  assert(src.isEmpty());

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  const size_t channels = src.channels();
  assert(rows > 0 && cols > 0 && channels > 0);
  // Deep copy
  dst = src;

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    return;
  }

  /* Compute Gaussian Kernal */
  const int half_size_kernal = std::ceil(3.0 * sigma);
  Eigen::VectorXd kernal(2 * half_size_kernal + 1);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal(i + half_size_kernal) =
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma)); // * (1.0 + (i + half_size_kernal) / 1000.0); //
                               // util::computeGaussian(i, sigma, 0.0);
  }
  kernal /= kernal.sum();
  // std::cout << "kernal: " << kernal.transpose() << std::endl;

  /* Convolve in x direction */
  Image tmp_dst(rows, cols, channels);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = util::clamp<size_t>(col + idx, 0, cols - 1);
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal(half_size_kernal + idx) *
                         static_cast<double>(src.at(row, c, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        tmp_dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp<double>(blurred(ch), 0, 255));
      }
    }
  }

  /* Convolve in y direction */
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = util::clamp<size_t>(row + idx, 0, rows - 1);
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal(half_size_kernal + idx) *
                         static_cast<double>(tmp_dst.at(r, col, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp<double>(blurred(ch), 0, 255));
      }
    }
  }
}

void blurGaussianNew(const Image &src, Image &dst, const double &sigma) {
  assert(sigma > 0);
  assert(src.isEmpty());

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  const size_t channels = src.channels();
  assert(rows > 0 && cols > 0 && channels > 0);
  // Deep copy
  dst = src;

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    return;
  }

  /* Compute Gaussian Kernal */
  const int half_size_kernal = std::ceil(3.0 * sigma);
  Eigen::VectorXd kernal(2 * half_size_kernal + 1);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal(i + half_size_kernal) =
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma)); // * (1.0 + (i + half_size_kernal) / 1000.0); //
                               // util::computeGaussian(i, sigma, 0.0);
  }
  kernal /= kernal.sum();
  // std::cout << "kernal: " << kernal.transpose() << std::endl;

  /* Convolve in x direction */
  Image tmp_dst(rows, cols, channels);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = util::clamp<size_t>(col + idx, 0, cols - 1);
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal(half_size_kernal + idx) *
                         static_cast<double>(src.at(row, c, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        tmp_dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp<double>(blurred(ch), 0, 255));
      }
    }
  }

  /* Convolve in y direction */
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = util::clamp<size_t>(row + idx, 0, rows - 1);
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal(half_size_kernal + idx) *
                         static_cast<double>(tmp_dst.at(r, col, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp<double>(blurred(ch), 0, 255));
      }
    }
  }
}

void blurGaussian(const Image &src, EigenVec<Eigen::MatrixXd> &dst,
                  const double &sigma) {
  assert(sigma > 0);

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  const size_t channels = src.channels();
  assert(rows > 0 && cols > 0 && channels > 0);
  dst.clear();
  dst.resize(channels, Eigen::MatrixXd::Zero(rows, cols));

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    return;
  }

  /* Compute Gaussian Kernal */
  const int half_size_kernal = std::ceil(3.0 * sigma);
  Eigen::VectorXd kernal(2 * half_size_kernal + 1);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal(i + half_size_kernal) =
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma)); // * (1.0 + (i + half_size_kernal) / 1000.0);
                               // util::computeGaussian(i, sigma, 0.0);
  }
  kernal /= kernal.sum();
  // std::cout << "kernal: " << kernal.transpose() << std::endl;

  /* Convolve in x direction */
  EigenVec<Eigen::MatrixXd> tmp_dst = dst;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = util::clamp<size_t>(col + idx, 0, cols - 1);
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal(half_size_kernal + idx) *
                         static_cast<double>(src.at(row, c, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        tmp_dst[ch].operator()(row, col) = blurred(ch);
      }
    }
  }

  /* Convolve in y direction */
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = util::clamp<size_t>(row + idx, 0, rows - 1);
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) +=
              kernal(half_size_kernal + idx) * tmp_dst[ch].operator()(r, col);
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        dst[ch].operator()(row, col) = blurred(ch);
      }
    }
  }
}

void blurGaussian(const Eigen::MatrixXd &src, Eigen::MatrixXd &dst,
                  const double &sigma) {
  assert(sigma > 0);

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  assert(rows > 0 && cols > 0);

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    return;
  }

  /* Compute Gaussian Kernal */
  const int half_size_kernal =
      3.0 * sigma > 1.0 ? (int)std::ceil(3.0 * sigma) : 1;
  Eigen::VectorXd kernal(2 * half_size_kernal + 1);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal(i + half_size_kernal) =
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma)); // * (1.0 + (i + half_size_kernal) / 1000.0);
                               // util::computeGaussian(i, sigma, 0.0);
  }
  kernal /= kernal.sum();
  // std::cout << "kernal: " << kernal.transpose() << std::endl;

  /* Convolve in x direction */
  Eigen::MatrixXd tmp_dst = Eigen::MatrixXd::Zero(rows, cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = util::clamp<size_t>(col + idx, 0, cols - 1);
        tmp_dst(row, col) +=
            kernal(half_size_kernal + idx) * static_cast<double>(src(row, c));
      }
    }
  }

  /* Convolve in y direction */
  dst = Eigen::MatrixXd::Zero(rows, cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = util::clamp<size_t>(row + idx, 0, rows - 1);
        dst(row, col) += kernal(half_size_kernal + idx) * tmp_dst(r, col);
      }
    }
  }
}

void scaleImage(const Image &src, Image &dst, const double &scale) {
  assert(scale > std::numeric_limits<double>::epsilon());

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  const size_t channels = src.channels();
  const size_t scaled_rows = std::floor(rows * scale);
  const size_t scaled_cols = std::floor(cols * scale);
  assert(rows > 0 && cols > 0 && channels > 0);

  Image tmp_dst(scaled_rows, scaled_cols, channels);
  if (scaled_rows == 0 || scaled_cols == 0) {
    dst = tmp_dst;
    return;
  }

  for (size_t row = 0; row < scaled_rows; ++row) {
    const double row_src = util::clamp<double>(row / scale, 0, rows - 1.0);
    const size_t row_src1 = util::clamp<size_t>(row_src, 0U, rows - 1U);
    const size_t row_src2 = util::clamp<size_t>(row_src + 1, 0U, rows - 1U);
    ;
    for (size_t col = 0; col < scaled_cols; ++col) {
      const double col_src = util::clamp<double>(col / scale, 0, cols - 1.0);
      const size_t col_src1 = util::clamp<size_t>(col_src, 0U, cols - 1U);
      const size_t col_src2 = util::clamp<size_t>(col_src + 1, 0U, cols - 1U);
      ;
      for (size_t channel = 0; channel < channels; ++channel) {
        const double x11 =
            static_cast<double>(src.at(row_src1, col_src1, channel));
        const double x12 =
            static_cast<double>(src.at(row_src1, col_src2, channel));
        const double x21 =
            static_cast<double>(src.at(row_src2, col_src1, channel));
        const double x22 =
            static_cast<double>(src.at(row_src2, col_src2, channel));
        tmp_dst.at(row, col, channel) = static_cast<uint8_t>(util::clamp<int>(
            util::computeBilinearInterpolation(x11, x12, x21, x22, col_src1,
                                               col_src2, row_src1, row_src2,
                                               col_src, row_src),
            0, 255));
      }
    }
  }

  dst = tmp_dst;
}

EigenVec<Eigen::MatrixXd>
computeChannelWiseImageDifference(const Image &image1, const Image &image2) {
  assert(image1.rows() == image2.rows());
  assert(image1.cols() == image2.cols());
  assert(image1.channels() == image2.channels());

  const size_t rows = image1.rows();
  const size_t cols = image1.cols();
  const size_t channels = image1.channels();

  EigenVec<Eigen::MatrixXd> chw_diffs;
  if (rows == 0 || cols == 0 || channels == 0) {
    return chw_diffs;
  }

  chw_diffs.resize(channels);
  for (size_t channel = 0; channel < channels; ++channel) {
    Eigen::MatrixXd &diff = chw_diffs[channel];
    diff = Eigen::MatrixXd::Zero(rows, cols);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        diff(row, col) = static_cast<double>(image1.at(row, col, channel)) -
                         static_cast<double>(image2.at(row, col, channel));
      }
    }
  }

  return chw_diffs;
}

void convertImageColorBGR2GrayScale(const Image &src, Image &dst) {
  assert(src.channels() == 3);

  const size_t rows = src.rows();
  const size_t cols = src.cols();

  Image tmp_dst(rows, cols, 1);
  if (rows == 0 || cols == 0) {
    dst = tmp_dst;
    return;
  }

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      const auto bgr = src.getChannel(row, col);
      // tmp_dst.at(row, col, 0) =
      //     static_cast<uint8_t>(0.114 * bgr(0) + 0.587 * bgr(1) + 0.299 *
      //     bgr(2));
      double gray =
          (double(bgr(0)) + double(bgr(1)) + double(bgr(2))) / 3.0 + 0.5;
      gray = util::clamp<double>(gray, 0.0, 255.0);
      tmp_dst.at(row, col, 0) = static_cast<uint8_t>(gray);
    }
  }
  dst = tmp_dst;
}

void drawCircle(const base::Image &src, base::Image &dst,
                const double &center_x, const double &center_y,
                const double &radius, const uint8_t color[3],
                const double scale, const double thickness) {
  assert(radius >= 0);
  const size_t channels = src.channels();
  if (channels != 1 && channels != 3) {
    return;
  }
  const size_t rows = src.rows();
  const size_t cols = src.cols();

  dst = src;
  if (radius <= std::numeric_limits<double>::epsilon()) {
    if (static_cast<int>(center_x) >= 0 &&
        static_cast<int>(center_x) <= static_cast<int>(cols) - 1 &&
        static_cast<int>(center_y) >= 0 &&
        static_cast<int>(center_y) <= static_cast<int>(rows) - 1) {
      if (channels == 1) {
        dst.at(center_y, center_x, 0) = static_cast<uint8_t>(
            0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]);
      } else if (channels == 3) {
        for (int i = 0; i < 3; ++i) {
          dst.at(center_y, center_x, i) = color[i];
        }
      }
    }
    return;
  }

  const double scaled_radius = radius * scale;
  const double angle_step = 1.0 / (scaled_radius + thickness);
  for (double angle = 0.0; angle < 2.0 * M_PI; angle += angle_step) {
    for (double dr = -thickness * 0.5; dr <= thickness * 0.5; ++dr) {
      const int x =
          static_cast<int>(center_x + (scaled_radius + dr) * std::cos(angle));
      const int y =
          static_cast<int>(center_y + (scaled_radius + dr) * std::sin(angle));
      if (x >= 0 && x <= static_cast<int>(cols) - 1 && y >= 0 &&
          y <= static_cast<int>(rows) - 1) {
        if (channels == 1) {
          dst.at(y, x, 0) = static_cast<uint8_t>(
              0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]);
        } else if (channels == 3) {
          for (int i = 0; i < 3; ++i) {
            dst.at(y, x, i) = color[i];
          }
        }
      }
    }
  }
}

void drawLine(const base::Image &src, base::Image &dst,
              const Eigen::Vector2d &point1, const Eigen::Vector2d &point2,
              const uint8_t color[3], const double thickness) {
  const size_t channels = src.channels();
  if (channels != 1 && channels != 3) {
    return;
  }
  const size_t rows = src.rows();
  const size_t cols = src.cols();

  dst = src;
  // First draw the end points
  const int x1 = static_cast<int>(point1.x());
  const int y1 = static_cast<int>(point1.y());
  const int x2 = static_cast<int>(point2.x());
  const int y2 = static_cast<int>(point2.y());
  if (x1 >= 0 && x1 <= static_cast<int>(cols) - 1 && y1 >= 0 &&
      y1 <= static_cast<int>(rows) - 1) {
    if (channels == 1) {
      dst.at(y1, x1, 0) = static_cast<uint8_t>(
          0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]);
    } else if (channels == 3) {
      for (int i = 0; i < 3; ++i) {
        dst.at(y1, x1, i) = color[i];
      }
    }
  }
  if (x2 >= 0 && x2 <= static_cast<int>(cols) - 1 && y2 >= 0 &&
      y2 <= static_cast<int>(rows) - 1) {
    if (channels == 1) {
      dst.at(y2, x2, 0) = static_cast<uint8_t>(
          0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]);
    } else if (channels == 3) {
      for (int i = 0; i < 3; ++i) {
        dst.at(y2, x2, i) = color[i];
      }
    }
  }
  // Then draw the points between end points
  // TODO Add thickness property
  const Eigen::Vector2d &direction = (point2 - point1).normalized();
  const double length = (point2 - point1).norm();
  // std::cout << point1.transpose() << " " << point2.transpose() <<
  // direction.norm() << " " << (point2 - point1).norm() << std::endl;
  if (std::abs(direction.norm() - 1.0) <= 1e-5) {
    for (double t = 0.0; t <= length; ++t) {
      const int x = static_cast<int>(point1.x() + t * direction.x());
      const int y = static_cast<int>(point1.y() + t * direction.y());
      if (x >= 0 && x <= static_cast<int>(cols) - 1 && y >= 0 &&
          y <= static_cast<int>(rows) - 1) {
        // std::cout << "drawn " << x << ", " << y << std::endl;
        if (channels == 1) {
          dst.at(y, x, 0) = static_cast<uint8_t>(
              0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]);
        } else if (channels == 3) {
          for (int i = 0; i < 3; ++i) {
            dst.at(y, x, i) = color[i];
          }
        }
      }
    }
  }
}

} // namespace base
} // namespace my3d
