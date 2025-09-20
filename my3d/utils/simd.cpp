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

__m256i avx_min_epi64(__m256i const &a, __m256i const &b) {
  __m256i mask = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(b, a, mask);
}

__m256i avx_max_epi64(__m256i const &a, __m256i const &b) {
  __m256i mask = _mm256_cmpgt_epi64(b, a);
  return _mm256_blendv_epi8(b, a, mask);
}

__m256i avx_clamp_epi32(__m256i const &a, __m256i const &l, __m256i const &u) {
  return _mm256_min_epi32(u, _mm256_max_epi32(l, a));
}

#endif // __AVX__

} // namespace simd
} // namespace util
} // namespace my3d
