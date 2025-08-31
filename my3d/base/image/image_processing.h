/*
 * filename: image_processing.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:  Some useful DIP algorithms.
 */

#ifndef _MY3D_BASE_IMAGE_IMAGE_PROCESSING_H_
#define _MY3D_BASE_IMAGE_IMAGE_PROCESSING_H_

#include "base/image/Image.h"
#include "utils/math.h"

namespace my3d {
namespace base {

/**
 * @brief Blur an image using a (separated) Gaussian kernal
 *
 * @param src       the source image
 * @param dst[out]  the blurred image
 * @param sigma     standard deviation of the Gaussian kernal
 */
void blurGaussian(const Image &src, Image &dst, const double &sigma);

/**
 * @brief Blur an image using a (separated) Gaussian kernal 
 *TODO: Delete this function before merging
 * @param src       the source image
 * @param dst[out]  the blurred image
 * @param sigma     standard deviation of the Gaussian kernal
 */
void blurGaussianNew(const Image &src, Image &dst, const double &sigma);

/**
 * @brief Blur an image using a (separated) Gaussian kernal and
 * use Eigen::MatrixXd matrices to store the result (each matrix
 * represents a channel)
 *
 * @param src       the source image
 * @param dst[out]  the blurred image
 * @param sigma     standard deviation of the Gaussian kernal
 */
void blurGaussian(const Image &src, EigenVec<Eigen::MatrixXd> &dst,
                  const double &sigma);

/**
 * @brief Blur an image (represented by Eigen Matrix) using a (separated)
 * Gaussian kernal and use Eigen::MatrixXd matrices to store the result (each
 * matrix represents a channel)
 *
 * @param src       the source image
 * @param dst[out]  the blurred image
 * @param sigma     standard deviation of the Gaussian kernal
 */
void blurGaussian(const Eigen::MatrixXd &src, Eigen::MatrixXd &dst,
                  const double &sigma);

/**
 * @brief Scale an image
 *
 * @param src       the source image
 * @param dst[out]  the scaled image
 * @param scale     scaling factor (needs to be positive)
 */
void scaleImage(const Image &src, Image &dst, const double &scale);

/**
 * @brief Compute channel-wise difference between two images
 *
 * @param image1     the first image
 * @param image2     the second image
 *
 * @retval channel-wise difference matrices (stored by Eigen::MatrixXd)
 */
EigenVec<Eigen::MatrixXd>
computeChannelWiseImageDifference(const Image &image1, const Image &image2);

/**
 * @brief Convert a BGR image to a gray-scale image
 *
 *  @param src       the source BGR image
 *  @param dst[out]  the result gray-scale image
 */
void convertImageColorBGR2GrayScale(const Image &src, Image &dst);

/**
 * @brief Draw a circle on the image
 */
void drawCircle(const base::Image &src, base::Image &dst,
                const double &center_x, const double &center_y,
                const double &radius, const uint8_t color[3],
                const double scale = 1.0, const double thickness = 1.0);

/**
 * @brief Draw a line on the image
 *
 * TODO Add thickness property
 */
void drawLine(const base::Image &src, base::Image &dst,
              const Eigen::Vector2d &point1, const Eigen::Vector2d &point2,
              const uint8_t color[3], const double thickness = 1.0);

} // namespace base

} // namespace my3d

#endif // _MY3D_BASE_IMAGE_IMAGE_PROCESSING_H_
