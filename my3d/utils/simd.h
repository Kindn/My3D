/*
 * filename: simd.h
 * author:   Peiyan Liu
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_UTIL_SIMD_H_
#define _MY3D_UTIL_SIMD_H_

#ifdef __AVX__
#include <immintrin.h>
#endif // __AVX__

#ifdef MSVC
#define ALIGNED(N) __declspec(align(N))
#elif defined(__GNUC__)
#define ALIGNED(N) __attribute__((__aligned__(N)))
#endif

namespace my3d {
namespace util {
namespace simd {

#ifdef __AVX__
static __m256i const first_n_mask_32[9]{
    _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0),
    _mm256_setr_epi32(~0, 0, 0, 0, 0, 0, 0, 0),
    _mm256_setr_epi32(~0, ~0, 0, 0, 0, 0, 0, 0),
    _mm256_setr_epi32(~0, ~0, ~0, 0, 0, 0, 0, 0),
    _mm256_setr_epi32(~0, ~0, ~0, ~0, 0, 0, 0, 0),
    _mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, 0, 0, 0),
    _mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, 0, 0),
    _mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, 0),
    _mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0)};

static __m256i const first_n_mask_64[5]{
    _mm256_setr_epi64x(0, 0, 0, 0), _mm256_setr_epi64x(~0, 0, 0, 0),
    _mm256_setr_epi64x(~0, ~0, 0, 0), _mm256_setr_epi64x(~0, ~0, ~0, 0),
    _mm256_setr_epi64x(~0, ~0, ~0, ~0)};

__m256 avx_load_first_n_ps(float const *p, size_t const n);

__m256d avx_load_first_n_pd(double const *p, size_t const n);

__m256i avx_load_first_n_epi32(int32_t const *p, size_t const n);
#endif // __AVX__

} // namespace simd
} // namespace util
} // namespace my3d

#endif // _MY3D_UTIL_SIMD_H_