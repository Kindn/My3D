/*
 * filename: simd.cpp
 * author:   Peiyan Liu
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "utils/simd.h"

namespace my3d {
namespace util {
namespace simd {

#ifdef __AVX__
__m256 avx_load_first_n_ps(float const *p, size_t const n) {
  __m256i const mask{first_n_mask_32[n > 8U ? 8U : n]};
  return _mm256_maskload_ps(p, mask);
}

__m256d avx_load_first_n_pd(double const *p, size_t const n) {
  __m256i const mask{first_n_mask_64[n > 4U ? 4U : n]};
  return _mm256_maskload_pd(p, mask);
}

__m256i avx_load_first_n_epi32(int32_t const *p, size_t const n) {
  __m256i const mask{first_n_mask_64[n > 4U ? 4U : n]};
  return _mm256_maskload_epi32(p, mask);
}

#endif // __AVX__

} // namespace simd
} // namespace util
} // namespace my3d
