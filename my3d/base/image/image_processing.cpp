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
  assert(!src.isEmpty());

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
  const int64_t half_size_kernal = static_cast<int64_t>(std::ceil(3.0 * sigma));
  size_t const kernal_size{2UL * half_size_kernal + 1UL};
  float *const kernal{
      (float *)aligned_alloc(32UL, sizeof(float) * kernal_size)};
  Eigen::Map<Eigen::VectorXf> kernal_map(kernal, kernal_size);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal_map(i + half_size_kernal) =
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma)); // * (1.0 + (i + half_size_kernal) / 1000.0); //
                               // util::computeGaussian(i, sigma, 0.0);
  }
  kernal_map /= kernal_map.sum();

#ifdef __AVX__
  size_t const kNumRemKernalValues{kernal_size % 8UL};
  size_t const kNumKernalBlocks{kernal_size / 8UL +
                                ((kNumRemKernalValues == 0) ? 0UL : 1UL)};
  __m256i const kernal_offsets0{_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};
  __m256i const kernal_steps{_mm256_set1_epi32(8)};
#endif // __AVX__

  /* Convolve in x direction */
  size_t const kStepRow{cols * channels};
  size_t const kStepCol{channels};
  Image tmp_dst(rows, cols, channels);
  uint32_t const num_threads{std::thread::hardware_concurrency()};
  omp_set_num_threads(num_threads);
#pragma omp parallel for
  for (size_t row = 0UL; row < rows; ++row) {
    for (size_t col{0UL}; col < cols; ++col) {
#ifndef __AVX__
      Eigen::Matrix<float, Eigen::Dynamic, 1> blurred{
          Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(channels)};
      for (int64_t idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = static_cast<size_t>(static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(col) + idx, 0,
                                 static_cast<int64_t>(cols) - 1)));
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal[static_cast<int>(half_size_kernal) + idx] *
                         static_cast<float>(src.at(row, c, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        tmp_dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp<float>(blurred(ch), 0, 255));
      }
#else  // __AVX__
      float const *const pk{kernal};
      __m256i const c0{_mm256_set1_epi32(
          static_cast<int32_t>(col) - static_cast<int32_t>(half_size_kernal))};
      for (size_t ch{0UL}; ch < channels; ++ch) {
        __m256 sum_vec{_mm256_setzero_ps()};
        size_t kernal_block_idx{0UL};
        __m256i offsets_vec{_mm256_add_epi32(kernal_offsets0, c0)};
        ALIGNED(32) int32_t offsets[8]{0};
        for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 8L) {
          _mm256_store_si256((__m256i *)&offsets[0], offsets_vec);
          // __m256i const kernal_block{kernal_blocks[kernal_block_idx]};
          size_t const num_loaded{(i + 8UL > kernal_size) ? kNumRemKernalValues
                                                          : 8UL};
          __m256 const kernal_block =
              util::simd::avx_load_first_n_ps(pk + i, num_loaded);
          int64_t const kMaxCol{static_cast<int64_t>(cols) - 1L};
          size_t const cs[8]{
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[0], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[1], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[2], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[3], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[4], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[5], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[6], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[7], 0L, kMaxCol)),
          };
          size_t const start{row * kStepRow + ch};
          uint8_t const *ptr_src{src.data() + start};
          ALIGNED(32)
          float const pixels[8]{(float)ptr_src[cs[0] * kStepCol],
                                (float)ptr_src[cs[1] * kStepCol],
                                (float)ptr_src[cs[2] * kStepCol],
                                (float)ptr_src[cs[3] * kStepCol],
                                (float)ptr_src[cs[4] * kStepCol],
                                (float)ptr_src[cs[5] * kStepCol],
                                (float)ptr_src[cs[6] * kStepCol],
                                (float)ptr_src[cs[7] * kStepCol]};
          __m256 const image_block{_mm256_loadu_ps(pixels)};
          __m256 const weighted_pixels{
              _mm256_mul_ps(kernal_block, image_block)};
          sum_vec = _mm256_add_ps(weighted_pixels, sum_vec);
          ++kernal_block_idx;
          offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
        }
        ALIGNED(32) float sum[8]{0.0f};
        _mm256_store_ps(sum, sum_vec);
        float blurred{(sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] +
                       sum[6] + sum[7])};
        tmp_dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp(blurred, 0.0f, 255.0f));
      }
#endif // __AVX__
    }
  }

  /* Convolve in y direction */
#pragma omp parallel for
  for (size_t row = 0UL; row < rows; ++row) {
#ifdef __AVX__
    __m256i const r0{_mm256_set1_epi32(static_cast<int32_t>(row) -
                                       static_cast<int32_t>(half_size_kernal))};
#endif // __AVX__
    for (size_t col{0UL}; col < cols; ++col) {
#ifndef __AVX__
      Eigen::Matrix<float, Eigen::Dynamic, 1> blurred{
          Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(channels)};
      for (int64_t idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(col) + idx, 0,
                                 static_cast<int64_t>(cols) - 1));
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal[static_cast<int>(half_size_kernal) + idx] *
                         static_cast<float>(tmp_dst.at(r, col, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp<float>(blurred(ch), 0, 255));
      }
#else  // __AVX__
      float const *const pk{kernal};
      for (size_t ch{0UL}; ch < channels; ++ch) {
        __m256 sum_vec{_mm256_setzero_ps()};
        size_t kernal_block_idx{0UL};
        __m256i offsets_vec{_mm256_add_epi32(kernal_offsets0, r0)};
        ALIGNED(32) int32_t offsets[8]{0};
        for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 8L) {
          // __m256i const kernal_block{kernal_blocks[kernal_block_idx]};
          size_t const num_loaded{(i + 8UL > kernal_size) ? kNumRemKernalValues
                                                          : 8UL};
          __m256 const kernal_block =
              util::simd::avx_load_first_n_ps(pk + i, num_loaded);
          _mm256_store_si256((__m256i *)&offsets[0], offsets_vec);
          int64_t const kMaxRow{static_cast<int64_t>(rows) - 1L};
          // __m256i const image_block{};
          // size_t const rs[8]{0UL};
          size_t const rs[8]{
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[0], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[1], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[2], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[3], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[4], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[5], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[6], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[7], 0L, kMaxRow)),
          };
          size_t const start{col * kStepCol + ch};
          uint8_t const *ptr_src{tmp_dst.data() + start};
          ALIGNED(32)
          float const pixels[8]{(float)ptr_src[rs[0] * kStepRow],
                                (float)ptr_src[rs[1] * kStepRow],
                                (float)ptr_src[rs[2] * kStepRow],
                                (float)ptr_src[rs[3] * kStepRow],
                                (float)ptr_src[rs[4] * kStepRow],
                                (float)ptr_src[rs[5] * kStepRow],
                                (float)ptr_src[rs[6] * kStepRow],
                                (float)ptr_src[rs[7] * kStepRow]};
          __m256 const image_block{_mm256_loadu_ps(pixels)};
          __m256 const weighted_pixels{
              _mm256_mul_ps(kernal_block, image_block)};
          sum_vec = _mm256_add_ps(weighted_pixels, sum_vec);
          ++kernal_block_idx;
          offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
        }
        ALIGNED(32) float sum[8]{0.0f};
        _mm256_store_ps(sum, sum_vec);
        float blurred{(sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] +
                       sum[6] + sum[7])};
        dst.at(row, col, ch) =
            static_cast<uint8_t>(util::clamp(blurred, 0.0f, 255.0f));
      }
#endif // __AVX__
    }
  }
  std::free(kernal);
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
  const int64_t half_size_kernal = static_cast<int64_t>(std::ceil(3.0 * sigma));
  size_t const kernal_size{2UL * half_size_kernal + 1UL};
  float *const kernal{
      (float *)aligned_alloc(32UL, sizeof(float) * kernal_size)};
  Eigen::Map<Eigen::VectorXf> kernal_map(kernal, kernal_size);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal_map(i + half_size_kernal) =
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma)); // * (1.0 + (i + half_size_kernal) / 1000.0); //
                               // util::computeGaussian(i, sigma, 0.0);
  }
  kernal_map /= kernal_map.sum();

#ifdef __AVX__
  size_t const kNumRemKernalValues{kernal_size % 8UL};
  size_t const kNumKernalBlocks{kernal_size / 8UL +
                                ((kNumRemKernalValues == 0) ? 0UL : 1UL)};
  __m256i const kernal_offsets0{_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};
  __m256i const kernal_steps{_mm256_set1_epi32(8)};
#endif // __AVX__

  /* Convolve in x direction */
  size_t const kStepRow{cols * channels};
  size_t const kStepCol{channels};
  EigenVec<Eigen::MatrixXd> tmp_dst(channels,
                                    Eigen::MatrixXd::Zero(rows, cols));
  uint32_t const num_threads{std::thread::hardware_concurrency()};
  omp_set_num_threads(num_threads);
#pragma omp parallel for
  for (size_t row = 0UL; row < rows; ++row) {
    for (size_t col = 0UL; col < cols; ++col) {
#ifndef __AVX__
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(col) + idx, 0L,
                                 static_cast<int64_t>(cols) - 1L));
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal[static_cast<int>(half_size_kernal) + idx] *
                         static_cast<double>(src.at(row, c, ch));
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        tmp_dst[ch].operator()(row, col) = blurred(ch);
      }
#else  // __AVX__
      float const *const pk{kernal};
      __m256i const c0{_mm256_set1_epi32(
          static_cast<int32_t>(col) - static_cast<int32_t>(half_size_kernal))};
      for (size_t ch{0UL}; ch < channels; ++ch) {
        __m256 sum_vec{_mm256_setzero_ps()};
        size_t kernal_block_idx{0UL};
        __m256i offsets_vec{_mm256_add_epi32(kernal_offsets0, c0)};
        ALIGNED(32) int32_t offsets[8]{0};
        for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 8L) {
          _mm256_store_si256((__m256i *)&offsets[0], offsets_vec);
          // __m256i const kernal_block{kernal_blocks[kernal_block_idx]};
          size_t const num_loaded{(i + 8UL > kernal_size) ? kNumRemKernalValues
                                                          : 8UL};
          __m256 const kernal_block =
              util::simd::avx_load_first_n_ps(pk + i, num_loaded);
          int64_t const kMaxCol{static_cast<int64_t>(cols) - 1L};
          size_t const cs[8]{
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[0], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[1], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[2], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[3], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[4], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[5], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[6], 0L, kMaxCol)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[7], 0L, kMaxCol)),
          };
          size_t const start{row * kStepRow + ch};
          uint8_t const *ptr_src{src.data() + start};
          ALIGNED(32)
          float const pixels[8]{(float)ptr_src[cs[0] * kStepCol],
                                (float)ptr_src[cs[1] * kStepCol],
                                (float)ptr_src[cs[2] * kStepCol],
                                (float)ptr_src[cs[3] * kStepCol],
                                (float)ptr_src[cs[4] * kStepCol],
                                (float)ptr_src[cs[5] * kStepCol],
                                (float)ptr_src[cs[6] * kStepCol],
                                (float)ptr_src[cs[7] * kStepCol]};
          __m256 const image_block{_mm256_loadu_ps(pixels)};
          __m256 const weighted_pixels{
              _mm256_mul_ps(kernal_block, image_block)};
          sum_vec = _mm256_add_ps(weighted_pixels, sum_vec);
          ++kernal_block_idx;
          offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
        }
        ALIGNED(32) float sum[8]{0.0f};
        _mm256_store_ps(sum, sum_vec);
        float blurred{(sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] +
                       sum[6] + sum[7])};
        tmp_dst[ch].operator()(row, col) = blurred;
      }
#endif // __AVX__
    }
  }

  /* Convolve in y direction */
#pragma omp parallel for
  for (size_t col = 0UL; col < cols; ++col) {
    for (size_t row = 0UL; row < rows; ++row) {
#ifndef __AVX__
      Eigen::VectorXd blurred = Eigen::VectorXd::Zero(channels);
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(row) + idx, 0L,
                                 static_cast<int64_t>(rows) - 1L));
        for (size_t ch = 0; ch < channels; ++ch) {
          blurred(ch) += kernal[static_cast<int>(half_size_kernal) + idx] *
                         tmp_dst[ch].operator()(r, col);
        }
      }
      for (size_t ch = 0; ch < channels; ++ch) {
        dst[ch].operator()(row, col) = blurred(ch);
      }
#else  // __AVX__
      __m256i const r0{_mm256_set1_epi32(
          static_cast<int32_t>(row) - static_cast<int32_t>(half_size_kernal))};
      float const *const pk{kernal};
      for (size_t ch{0UL}; ch < channels; ++ch) {
        __m256 sum_vec{_mm256_setzero_ps()};
        size_t kernal_block_idx{0UL};
        __m256i offsets_vec{_mm256_add_epi32(kernal_offsets0, r0)};
        ALIGNED(32) int32_t offsets[8]{0};
        for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 8L) {
          // __m256i const kernal_block{kernal_blocks[kernal_block_idx]};
          size_t const num_loaded{(i + 8UL > kernal_size) ? kNumRemKernalValues
                                                          : 8UL};
          __m256 const kernal_block =
              util::simd::avx_load_first_n_ps(pk + i, num_loaded);
          _mm256_store_si256((__m256i *)&offsets[0], offsets_vec);
          int64_t const kMaxRow{static_cast<int64_t>(rows) - 1L};
          // __m256i const image_block{};
          // size_t const rs[8]{0UL};
          size_t const rs[8]{
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[0], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[1], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[2], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[3], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[4], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[5], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[6], 0L, kMaxRow)),
              static_cast<size_t>(
                  util::clamp<int64_t>(offsets[7], 0L, kMaxRow)),
          };
          //! Eigen matrices are stored in column major by default
          size_t const start{col * rows};
          double const *ptr_src{tmp_dst[ch].data() + start};
          ALIGNED(32)
          float const pixels[8]{(float)ptr_src[rs[0]], (float)ptr_src[rs[1]],
                                (float)ptr_src[rs[2]], (float)ptr_src[rs[3]],
                                (float)ptr_src[rs[4]], (float)ptr_src[rs[5]],
                                (float)ptr_src[rs[6]], (float)ptr_src[rs[7]]};
          __m256 const image_block{_mm256_loadu_ps(pixels)};
          __m256 const weighted_pixels{
              _mm256_mul_ps(kernal_block, image_block)};
          sum_vec = _mm256_add_ps(weighted_pixels, sum_vec);
          ++kernal_block_idx;
          offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
        }
        ALIGNED(32) float sum[8]{0.0f};
        _mm256_store_ps(sum, sum_vec);
        float blurred{(sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] +
                       sum[6] + sum[7])};
        dst[ch].operator()(row, col) = blurred;
      }
#endif // __AVX__
    }
  }
  std::free(kernal);
}

void blurGaussian(const Eigen::MatrixXd &src, Eigen::MatrixXd &dst,
                  const double &sigma) {
  assert(sigma > 0);

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  assert(rows > 0 && cols > 0);
  size_t const max_row_col{std::max(rows, cols)};

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    return;
  }

  /* Compute Gaussian Kernal */
  const int64_t half_size_kernal{
      3.0 * sigma > 1.0 ? static_cast<int64_t>(std::ceil(3.0 * sigma)) : 1L};
  size_t const kernal_size{2UL * half_size_kernal + 1UL};
  double *const kernal{
      (double *)aligned_alloc(32UL, sizeof(double) * kernal_size)};
  assert(kernal != nullptr);
  Eigen::Map<Eigen::VectorXd> kernal_map(kernal, kernal_size);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal_map(i + half_size_kernal) = static_cast<double>(
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma))); // * (1.0 + (i + half_size_kernal) / 1000.0); //
                                // util::computeGaussian(i, sigma, 0.0);
  }
  kernal_map /= kernal_map.sum();

  uint32_t const num_threads{std::thread::hardware_concurrency()};
  omp_set_num_threads(num_threads);

#ifdef __AVX__
  size_t const kNumRemKernalValues{kernal_size % 4UL};
  size_t const kNumKernalBlocks{kernal_size / 4UL +
                                ((kNumRemKernalValues == 0) ? 0UL : 1UL)};
  // __m256 * const kernal_blocks{(__m256 *)aligned_alloc(32UL, sizeof(__m256) *
  // kNumKernalBlocks)}; for (int32_t i{0L}; i <
  // static_cast<int64_t>(kernal_size); ++i) {
  //   kernal_blocks[i] = util::simd::avx_load_first_n_ps(
  //       kernal + i * 8, std::min(8, static_cast<int32_t>(kernal_size) - i *
  //       8));
  // }
  // std::cout << "aaaaaa" << std::endl;
  __m256i const kernal_offsets0{_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};
  __m256i const kernal_steps{_mm256_set1_epi32(4)};
  __m256i const min_row{_mm256_set1_epi32(0)};
  __m256i const max_row{_mm256_set1_epi32(static_cast<int32_t>(rows) - 1)};
  __m256i const min_col{_mm256_set1_epi32(0)};
  __m256i const max_col{_mm256_set1_epi32(static_cast<int32_t>(cols) - 1)};
  __m256i *const offsets_vecs{
      (__m256i *)aligned_alloc(32UL, sizeof(__m256i) * max_row_col)};
#pragma omp parallel for
  for (size_t i = 0UL; i < max_row_col; ++i) {
    offsets_vecs[i] = _mm256_add_epi32(
        kernal_offsets0,
        _mm256_set1_epi32(static_cast<int32_t>(i) -
                          static_cast<int32_t>(half_size_kernal)));
  }
#endif // __AVX__

  /* Convolve in x direction */
  double *const ptr_tmp_dst{
      (double *)aligned_alloc(32UL, sizeof(double) * (rows * cols))};
  // std::cout << ptr_tmp_dst << std::endl;
  assert(ptr_tmp_dst != nullptr);
  Eigen::Map<Eigen::MatrixXd> tmp_dst(ptr_tmp_dst, rows, cols);

#pragma omp parallel for
  for (size_t col = 0UL; col < cols; ++col) {
    for (size_t row = 0UL; row < rows; ++row) {
#ifndef __AVX__
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(col) + idx, 0L,
                                 static_cast<int64_t>(cols) - 1L));
        tmp_dst(row, col) += kernal[static_cast<int>(half_size_kernal) + idx] *
                             static_cast<float>(src(row, c));
      }
#else // __AVX__
      double const *const pk{kernal};
      __m256d sum_vec{_mm256_setzero_pd()};
      __m256i offsets_vec{offsets_vecs[col]};
      ALIGNED(32) int32_t offsets[8]{0};
      float blurred{0.0f};
      for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 4L) {
        _mm256_store_si256(
            (__m256i *)&offsets[0],
            util::simd::avx_clamp_epi32(offsets_vec, min_col, max_col));
        size_t const num_loaded{(i + 4UL > kernal_size) ? kNumRemKernalValues
                                                        : 4UL};
        __m256d const kernal_block =
            util::simd::avx_load_first_n_pd(pk + i, num_loaded);
        // __m256 const kernal_block{kernal_blocks[i / 8L]};
        __m256d const image_block{
            _mm256_setr_pd(src(row, offsets[0]), src(row, offsets[1]),
                           src(row, offsets[2]), src(row, offsets[3]))};
        __m256d const weighted_pixels{_mm256_mul_pd(kernal_block, image_block)};
        sum_vec = _mm256_add_pd(weighted_pixels, sum_vec);
        offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
      }
      ALIGNED(32) double sum[8]{0.0f};
      _mm256_storeu_pd(sum, sum_vec);
      tmp_dst(row, col) =
          sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

#endif // __AVX__
    }
  }

  /* Convolve in y direction */
  dst = Eigen::MatrixXd::Zero(rows, cols);
#pragma omp parallel for
  for (size_t col = 0UL; col < cols; ++col) {
    for (size_t row = 0UL; row < rows; ++row) {
#ifndef __AVX__
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(row) + idx, 0L,
                                 static_cast<int64_t>(rows) - 1L));
        dst(row, col) +=
            kernal[static_cast<int>(half_size_kernal) + idx] * tmp_dst(r, col);
      }
#else  // __AVX__
      double const *const pk{kernal};
      __m256d sum_vec{_mm256_setzero_pd()};
      __m256i offsets_vec{offsets_vecs[row]};
      double const *const &ptr_col{ptr_tmp_dst + col * rows};
      ALIGNED(32) int32_t offsets[8]{0};
      for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 4L) {
        _mm256_store_si256(
            (__m256i *)&offsets[0],
            util::simd::avx_clamp_epi32(offsets_vec, min_row, max_row));
        size_t const num_loaded{(i + 4UL > kernal_size) ? kNumRemKernalValues
                                                        : 4UL};
        __m256d const kernal_block =
            util::simd::avx_load_first_n_pd(pk + i, num_loaded);
        // __m256 const kernal_block{kernal_blocks[i / 8L]};
        int64_t const block_start_row{static_cast<int64_t>(row) -
                                      static_cast<int64_t>(half_size_kernal) +
                                      i};
        // float const * const block_start{ptr_col + block_start_row};
        __m256d const image_block{
            (block_start_row >= 0L &&
             block_start_row + 4L <= static_cast<int64_t>(rows))
                ? _mm256_loadu_pd(ptr_col + block_start_row)
                : _mm256_setr_pd(
                      tmp_dst(offsets[0], col), tmp_dst(offsets[1], col),
                      tmp_dst(offsets[2], col), tmp_dst(offsets[3], col))};
        __m256d const weighted_pixels{_mm256_mul_pd(kernal_block, image_block)};
        sum_vec = _mm256_add_pd(weighted_pixels, sum_vec);
        offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
      }
      ALIGNED(32) double sum[8]{0.0f};
      _mm256_store_pd(sum, sum_vec);
      dst(row, col) =
          sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif // __AVX__
    }
  }
  std::free(kernal);
  std::free(ptr_tmp_dst);
}

void blurGaussian(const Eigen::MatrixXf &src, Eigen::MatrixXf &dst,
                  const double &sigma) {
  assert(sigma > 0);

  const size_t rows = src.rows();
  const size_t cols = src.cols();
  assert(rows > 0 && cols > 0);
  size_t const max_row_col{std::max(rows, cols)};

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    return;
  }

  /* Compute Gaussian Kernal */
  const int64_t half_size_kernal{
      3.0 * sigma > 1.0 ? static_cast<int64_t>(std::ceil(3.0 * sigma)) : 1L};
  size_t const kernal_size{2UL * half_size_kernal + 1UL};
  float *const kernal{
      (float *)aligned_alloc(32UL, sizeof(float) * kernal_size)};
  assert(kernal != nullptr);
  Eigen::Map<Eigen::VectorXf> kernal_map(kernal, kernal_size);
  for (int i = -half_size_kernal; i <= half_size_kernal; ++i) {
    kernal_map(i + half_size_kernal) = static_cast<float>(
        std::exp(-0.5 * (i / sigma) *
                 (i / sigma))); // * (1.0 + (i + half_size_kernal) / 1000.0); //
                                // util::computeGaussian(i, sigma, 0.0);
  }
  kernal_map /= kernal_map.sum();

  uint32_t const num_threads{std::thread::hardware_concurrency()};
  omp_set_num_threads(num_threads);

#ifdef __AVX__
  size_t const kNumRemKernalValues{kernal_size % 8UL};
  size_t const kNumKernalBlocks{kernal_size / 8UL +
                                ((kNumRemKernalValues == 0) ? 0UL : 1UL)};
  // __m256 * const kernal_blocks{(__m256 *)aligned_alloc(32UL, sizeof(__m256) *
  // kNumKernalBlocks)}; for (int32_t i{0L}; i <
  // static_cast<int64_t>(kernal_size); ++i) {
  //   kernal_blocks[i] = util::simd::avx_load_first_n_ps(
  //       kernal + i * 8, std::min(8, static_cast<int32_t>(kernal_size) - i *
  //       8));
  // }
  // std::cout << "aaaaaa" << std::endl;
  __m256i const kernal_offsets0{_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};
  __m256i const kernal_steps{_mm256_set1_epi32(8)};
  __m256i const min_row{_mm256_set1_epi32(0)};
  __m256i const max_row{_mm256_set1_epi32(static_cast<int32_t>(rows) - 1)};
  __m256i const min_col{_mm256_set1_epi32(0)};
  __m256i const max_col{_mm256_set1_epi32(static_cast<int32_t>(cols) - 1)};
  __m256i *const offsets_vecs{
      (__m256i *)aligned_alloc(32UL, sizeof(__m256i) * max_row_col)};
#pragma omp parallel for
  for (size_t i = 0UL; i < max_row_col; ++i) {
    offsets_vecs[i] = _mm256_add_epi32(
        kernal_offsets0,
        _mm256_set1_epi32(static_cast<int32_t>(i) -
                          static_cast<int32_t>(half_size_kernal)));
  }
#endif // __AVX__

  /* Convolve in x direction */
  float *const ptr_tmp_dst{
      (float *)aligned_alloc(32UL, sizeof(float) * (rows * cols))};
  // std::cout << ptr_tmp_dst << std::endl;
  assert(ptr_tmp_dst != nullptr);
  Eigen::Map<Eigen::MatrixXf> tmp_dst(ptr_tmp_dst, rows, cols);

#pragma omp parallel for
  for (size_t col = 0UL; col < cols; ++col) {
    for (size_t row = 0UL; row < rows; ++row) {
#ifndef __AVX__
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t c = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(col) + idx, 0L,
                                 static_cast<int64_t>(cols) - 1L));
        tmp_dst(row, col) += kernal[static_cast<int>(half_size_kernal) + idx] *
                             static_cast<float>(src(row, c));
      }
#else // __AVX__
      float const *const pk{kernal};
      __m256 sum_vec{_mm256_setzero_ps()};
      __m256i offsets_vec{offsets_vecs[col]};
      ALIGNED(32) int32_t offsets[8]{0};
      float blurred{0.0f};
      for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 8L) {
        _mm256_store_si256(
            (__m256i *)&offsets[0],
            util::simd::avx_clamp_epi32(offsets_vec, min_col, max_col));
        size_t const num_loaded{(i + 8UL > kernal_size) ? kNumRemKernalValues
                                                        : 8UL};
        __m256 const kernal_block =
            util::simd::avx_load_first_n_ps(pk + i, num_loaded);
        // __m256 const kernal_block{kernal_blocks[i / 8L]};
        __m256 const image_block{_mm256_setr_ps(
            src(row, offsets[0]), src(row, offsets[1]), src(row, offsets[2]),
            src(row, offsets[3]), src(row, offsets[4]), src(row, offsets[5]),
            src(row, offsets[6]), src(row, offsets[7]))};
        __m256 const weighted_pixels{_mm256_mul_ps(kernal_block, image_block)};
        sum_vec = _mm256_add_ps(weighted_pixels, sum_vec);
        offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
      }
      ALIGNED(32) float sum[8]{0.0f};
      _mm256_storeu_ps(sum, sum_vec);
      tmp_dst(row, col) =
          sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

#endif // __AVX__
    }
  }

  /* Convolve in y direction */
  dst = Eigen::MatrixXf::Zero(rows, cols);
#pragma omp parallel for
  for (size_t col = 0UL; col < cols; ++col) {
    for (size_t row = 0UL; row < rows; ++row) {
#ifndef __AVX__
      for (int idx = -half_size_kernal; idx <= half_size_kernal; ++idx) {
        const size_t r = static_cast<size_t>(
            util::clamp<int64_t>(static_cast<int64_t>(row) + idx, 0L,
                                 static_cast<int64_t>(rows) - 1L));
        dst(row, col) +=
            kernal[static_cast<int>(half_size_kernal) + idx] * tmp_dst(r, col);
      }
#else  // __AVX__
      float const *const pk{kernal};
      __m256 sum_vec{_mm256_setzero_ps()};
      __m256i offsets_vec{offsets_vecs[row]};
      float const *const &ptr_col{ptr_tmp_dst + col * rows};
      ALIGNED(32) int32_t offsets[8]{0};
      for (int64_t i{0L}; i < static_cast<int64_t>(kernal_size); i += 8L) {
        _mm256_store_si256(
            (__m256i *)&offsets[0],
            util::simd::avx_clamp_epi32(offsets_vec, min_row, max_row));
        size_t const num_loaded{(i + 8UL > kernal_size) ? kNumRemKernalValues
                                                        : 8UL};
        __m256 const kernal_block =
            util::simd::avx_load_first_n_ps(pk + i, num_loaded);
        // __m256 const kernal_block{kernal_blocks[i / 8L]};
        int64_t const block_start_row{static_cast<int64_t>(row) -
                                      static_cast<int64_t>(half_size_kernal) +
                                      i};
        // float const * const block_start{ptr_col + block_start_row};
        __m256 const image_block{
            (block_start_row >= 0L &&
             block_start_row + 8L <= static_cast<int64_t>(rows))
                ? _mm256_loadu_ps(ptr_col + block_start_row)
                : _mm256_setr_ps(
                      tmp_dst(offsets[0], col), tmp_dst(offsets[1], col),
                      tmp_dst(offsets[2], col), tmp_dst(offsets[3], col),
                      tmp_dst(offsets[4], col), tmp_dst(offsets[5], col),
                      tmp_dst(offsets[6], col), tmp_dst(offsets[7], col))};
        __m256 const weighted_pixels{_mm256_mul_ps(kernal_block, image_block)};
        sum_vec = _mm256_add_ps(weighted_pixels, sum_vec);
        offsets_vec = _mm256_add_epi32(offsets_vec, kernal_steps);
      }
      ALIGNED(32) float sum[8]{0.0f};
      _mm256_store_ps(sum, sum_vec);
      dst(row, col) =
          sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif // __AVX__
    }
  }
  std::free(kernal);
  std::free(ptr_tmp_dst);
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

  for (size_t row = 0UL; row < scaled_rows; ++row) {
    const double row_src = util::clamp<double>(row / scale, 0, rows - 1.0);
    const size_t row_src1 = util::clamp<size_t>(row_src, 0U, rows - 1U);
    const size_t row_src2 = util::clamp<size_t>(row_src + 1, 0U, rows - 1U);
    ;
    for (size_t col = 0UL; col < scaled_cols; ++col) {
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
    for (size_t row = 0UL; row < rows; ++row) {
      for (size_t col = 0UL; col < cols; ++col) {
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

  for (size_t row = 0UL; row < rows; ++row) {
    for (size_t col = 0UL; col < cols; ++col) {
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
