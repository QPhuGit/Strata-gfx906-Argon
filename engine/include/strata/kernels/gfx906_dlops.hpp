// include/strata/kernels/gfx906_dlops.hpp
#pragma once

#include <cstdint>
#include <cmath>

#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>
#include <hip/hip_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace strata::kernels::gfx906 {

#if defined(__clang__) && (defined(__HIPCC__) || defined(__AMDGCN__))
typedef _Float16 f16x2_ext __attribute__((ext_vector_type(2)));
#endif

// ============================================================================
// 1. LỆNH PHẦN CỨNG v_dot4_i32_i8, v_dot4_u32_u8 & v_dot2_f32_f16 TRÊN GFX906
// ============================================================================

__device__ __forceinline__ int sdot4_i32_i8(const int a, const int b, const int c) {
#if (defined(__gfx906__) || defined(STRATA_HIP_GFX906)) && (defined(__HIP_DEVICE_COMPILE__) || defined(__AMDGCN__))
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    return c + int(pa[0]) * int(pb[0]) + int(pa[1]) * int(pb[1])
             + int(pa[2]) * int(pb[2]) + int(pa[3]) * int(pb[3]);
#endif
}

__device__ __forceinline__ uint32_t udot4_u32_u8(const uint32_t a, const uint32_t b, const uint32_t c) {
#if (defined(__gfx906__) || defined(STRATA_HIP_GFX906)) && (defined(__HIP_DEVICE_COMPILE__) || defined(__AMDGCN__))
    return __builtin_amdgcn_udot4(a, b, c, false);
#else
    const uint8_t* pa = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&b);
    return c + uint32_t(pa[0]) * uint32_t(pb[0]) + uint32_t(pa[1]) * uint32_t(pb[1])
             + uint32_t(pa[2]) * uint32_t(pb[2]) + uint32_t(pa[3]) * uint32_t(pb[3]);
#endif
}

__device__ __forceinline__ int udot4_u8_s8(const uint32_t a_u8, const int b_s8, const int c) {
    const uint32_t b_biased = static_cast<uint32_t>(b_s8) ^ 0x80808080u;
    const uint32_t prod_biased = udot4_u32_u8(a_u8, b_biased, 0u);
    const uint32_t sum_a = udot4_u32_u8(a_u8, 0x01010101u, 0u);
    return c + static_cast<int>(prod_biased) - (static_cast<int>(sum_a) << 7);
}

__device__ __forceinline__ int mode7_sdot4(const int g, const int m, const int u, const int acc) {
    const int pos_neg = sdot4_i32_i8(g ^ m, u, 0);
    const int bias_corr = sdot4_i32_i8(m, u, 0);
    return acc + (pos_neg - bias_corr);
}

__device__ __forceinline__ float fdot2_f32_f16(const __half2 a, const __half2 b, const float c) {
#if (defined(__gfx906__) || defined(STRATA_HIP_GFX906)) && defined(__HIP_DEVICE_COMPILE__)
    const f16x2_ext va = __builtin_bit_cast(f16x2_ext, a);
    const f16x2_ext vb = __builtin_bit_cast(f16x2_ext, b);
    return __builtin_amdgcn_fdot2(va, vb, c, false);
#else
    const float2 fa = __half22float2(a);
    const float2 fb = __half22float2(b);
    return fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, c));
#endif
}

// Gộp tổng 32 làn (logical warp) trên Wave64 bằng lệnh thanh ghi GFX9 DPP 1 chu kỳ
__device__ __forceinline__ float wave32_sum_gfx906(float val) {
#if (defined(__gfx906__) || defined(STRATA_HIP_GFX906)) && defined(__HIP_DEVICE_COMPILE__)
    val += __shfl_xor(val, 16, 32);
    val += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(val), 0x128, 0xF, 0xF, false)); // row_ror:8
    val += __shfl_xor(val, 4, 32);                                                                 // bước 4 bắt buộc dùng shfl_xor
    val += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(val), 0x4E,  0xF, 0xF, false)); // quad_perm:[2,3,0,1]
    val += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(val), 0xB1,  0xF, 0xF, false)); // quad_perm:[1,0,3,2]
    return val;
#else
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset, 32);
    return val;
#endif
}

// ============================================================================
// 2. DỊCH MŨ LŨY THỪA 2 CHÍNH XÁC & PHÂN RÃ KÉP (KHÔNG KẸP NGƯỠNG +-65504)
// ============================================================================

constexpr int kTargetUnbiasedExpFp16 = 13; // 2^13 = 8192.0 (nằm dưới 65504, giữ trọn 7-bit mantissa BF16)
constexpr int kLosslessExpSpanBf16   = 28; // Dải 28 bậc lũy thừa 2 không mất bit định trị BF16 trong FP16

__device__ __forceinline__ float bf16_bits_to_f32(const uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ void unpack_bf16x2_to_f32(const uint32_t bf16_pair, float& f0, float& f1) {
    f0 = __uint_as_float(bf16_pair << 16);
    f1 = __uint_as_float(bf16_pair & 0xFFFF0000u);
}

__device__ __forceinline__ int extract_unbiased_exp_f32(const float val, bool& is_zero_or_nonfinite) {
    const uint32_t bits = __float_as_uint(val) & 0x7FFFFFFFu;
    if (bits == 0u) {
        is_zero_or_nonfinite = true;
        return 0;
    }
    if (bits >= 0x7F800000u) {
        is_zero_or_nonfinite = true;
        return 128;
    }
    is_zero_or_nonfinite = false;
    const int biased_exp = static_cast<int>(bits >> 23);
    if (biased_exp == 0) {
        const int lz = __clz(bits) - 8;
        return -126 - lz;
    }
    return biased_exp - 127;
}

__device__ __forceinline__ float scale_pow2_f32(const float val, const int shift_exp) {
    if (val == 0.0f || shift_exp == 0) return val;
    const uint32_t bits = __float_as_uint(val);
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abs_bits = bits & 0x7FFFFFFFu;
    if (abs_bits >= 0x7F800000u) return val;

    int biased_exp = static_cast<int>(abs_bits >> 23);
    uint32_t mantissa = abs_bits & 0x007FFFFFu;

    if (biased_exp == 0) {
        const int lz = __clz(mantissa) - 8;
        mantissa = (mantissa << lz) & 0x007FFFFFu;
        biased_exp = 1 - lz;
    }

    const int new_biased_exp = biased_exp + shift_exp;
    if (new_biased_exp >= 255 || new_biased_exp <= 0) {
        return scalbnf(val, shift_exp);
    }
    return __uint_as_float(sign | (static_cast<uint32_t>(new_biased_exp) << 23) | mantissa);
}

struct DualScaledHalf2 {
    __half2 hi;
    __half2 lo;
    int shift_hi;
    int shift_lo;
    bool has_lo;
};

__device__ __forceinline__ DualScaledHalf2 bf16x2_to_dual_half2_exact(const uint32_t bf16_pair) {
    float f0, f1;
    unpack_bf16x2_to_f32(bf16_pair, f0, f1);

    bool z0 = false, z1 = false;
    const int e0 = extract_unbiased_exp_f32(f0, z0);
    const int e1 = extract_unbiased_exp_f32(f1, z1);

    DualScaledHalf2 res;
    res.has_lo = false;
    res.shift_lo = 0;
    res.lo = __floats2half2_rn(0.0f, 0.0f);

    if (z0 && z1) {
        res.shift_hi = 0;
        res.hi = __floats2half2_rn(f0, f1);
        return res;
    }
    if (z0) {
        res.shift_hi = kTargetUnbiasedExpFp16 - e1;
        res.hi = __floats2half2_rn(0.0f, scale_pow2_f32(f1, res.shift_hi));
        return res;
    }
    if (z1) {
        res.shift_hi = kTargetUnbiasedExpFp16 - e0;
        res.hi = __floats2half2_rn(scale_pow2_f32(f0, res.shift_hi), 0.0f);
        return res;
    }

    const int e_max = (e0 > e1) ? e0 : e1;
    const int e_min = (e0 < e1) ? e0 : e1;
    res.shift_hi = kTargetUnbiasedExpFp16 - e_max;

    if ((e_max - e_min) <= kLosslessExpSpanBf16) {
        res.hi = __floats2half2_rn(
            scale_pow2_f32(f0, res.shift_hi),
            scale_pow2_f32(f1, res.shift_hi)
        );
    } else {
        res.has_lo = true;
        res.shift_lo = kTargetUnbiasedExpFp16 - e_min;
        if (e0 > e1) {
            res.hi = __floats2half2_rn(scale_pow2_f32(f0, res.shift_hi), 0.0f);
            res.lo = __floats2half2_rn(0.0f, scale_pow2_f32(f1, res.shift_lo));
        } else {
            res.hi = __floats2half2_rn(0.0f, scale_pow2_f32(f1, res.shift_hi));
            res.lo = __floats2half2_rn(scale_pow2_f32(f0, res.shift_lo), 0.0f);
        }
    }
    return res;
}

__device__ __forceinline__ DualScaledHalf2 f32x2_to_dual_half2_exact(const float f0, const float f1) {
    bool z0 = false, z1 = false;
    const int e0 = extract_unbiased_exp_f32(f0, z0);
    const int e1 = extract_unbiased_exp_f32(f1, z1);

    DualScaledHalf2 res;
    res.has_lo = false;
    res.shift_lo = 0;
    res.lo = __floats2half2_rn(0.0f, 0.0f);

    if (z0 && z1) {
        res.shift_hi = 0;
        res.hi = __floats2half2_rn(f0, f1);
        return res;
    }
    if (z0) {
        res.shift_hi = kTargetUnbiasedExpFp16 - e1;
        res.hi = __floats2half2_rn(0.0f, scale_pow2_f32(f1, res.shift_hi));
        return res;
    }
    if (z1) {
        res.shift_hi = kTargetUnbiasedExpFp16 - e0;
        res.hi = __floats2half2_rn(scale_pow2_f32(f0, res.shift_hi), 0.0f);
        return res;
    }

    const int e_max = (e0 > e1) ? e0 : e1;
    const int e_min = (e0 < e1) ? e0 : e1;
    res.shift_hi = kTargetUnbiasedExpFp16 - e_max;

    if ((e_max - e_min) <= 14) {
        const float s0 = scale_pow2_f32(f0, res.shift_hi);
        const float s1 = scale_pow2_f32(f1, res.shift_hi);
        res.hi = __floats2half2_rn(s0, s1);
        const float2 hf = __half22float2(res.hi);
        const float r0 = s0 - hf.x, r1 = s1 - hf.y;
        if (r0 != 0.0f || r1 != 0.0f) {
            res.has_lo = true;
            res.shift_lo = res.shift_hi + 11;
            res.lo = __floats2half2_rn(scale_pow2_f32(r0, 11), scale_pow2_f32(r1, 11));
        }
    } else {
        res.has_lo = true;
        res.shift_lo = kTargetUnbiasedExpFp16 - e_min;
        if (e0 > e1) {
            res.hi = __floats2half2_rn(scale_pow2_f32(f0, res.shift_hi), 0.0f);
            res.lo = __floats2half2_rn(0.0f, scale_pow2_f32(f1, res.shift_lo));
        } else {
            res.hi = __floats2half2_rn(0.0f, scale_pow2_f32(f1, res.shift_hi));
            res.lo = __floats2half2_rn(scale_pow2_f32(f0, res.shift_lo), 0.0f);
        }
    }
    return res;
}

__device__ __forceinline__ float dot2_bf16x2_fdot2_exact(
    const uint32_t a_bf16x2,
    const uint32_t b_bf16x2,
    float acc
) {
    const DualScaledHalf2 da = bf16x2_to_dual_half2_exact(a_bf16x2);
    const DualScaledHalf2 db = bf16x2_to_dual_half2_exact(b_bf16x2);

    const float p_hihi = fdot2_f32_f16(da.hi, db.hi, 0.0f);
    acc += scale_pow2_f32(p_hihi, -(da.shift_hi + db.shift_hi));

    if (__builtin_expect(da.has_lo || db.has_lo, 0)) {
        if (da.has_lo) {
            const float p_lohi = fdot2_f32_f16(da.lo, db.hi, 0.0f);
            acc += scale_pow2_f32(p_lohi, -(da.shift_lo + db.shift_hi));
        }
        if (db.has_lo) {
            const float p_hilo = fdot2_f32_f16(da.hi, db.lo, 0.0f);
            acc += scale_pow2_f32(p_hilo, -(da.shift_hi + db.shift_lo));
        }
        if (da.has_lo && db.has_lo) {
            const float p_lolo = fdot2_f32_f16(da.lo, db.lo, 0.0f);
            acc += scale_pow2_f32(p_lolo, -(da.shift_lo + db.shift_lo));
        }
    }
    return acc;
}

__device__ __forceinline__ float dot2_bf16_pair_fdot2_exact(
    const uint16_t a0_bf16, const uint16_t a1_bf16,
    const uint16_t b0_bf16, const uint16_t b1_bf16,
    float acc
) {
    const uint32_t pa = static_cast<uint32_t>(a0_bf16) | (static_cast<uint32_t>(a1_bf16) << 16);
    const uint32_t pb = static_cast<uint32_t>(b0_bf16) | (static_cast<uint32_t>(b1_bf16) << 16);
    return dot2_bf16x2_fdot2_exact(pa, pb, acc);
}

__device__ __forceinline__ float dot2_bf16_f32_fdot2_exact(
    const uint32_t w_bf16x2,
    const float2 x_f32x2,
    float acc
) {
    const DualScaledHalf2 dw = bf16x2_to_dual_half2_exact(w_bf16x2);
    const DualScaledHalf2 dx = f32x2_to_dual_half2_exact(x_f32x2.x, x_f32x2.y);

    const float p_hihi = fdot2_f32_f16(dw.hi, dx.hi, 0.0f);
    acc += scale_pow2_f32(p_hihi, -(dw.shift_hi + dx.shift_hi));

    if (__builtin_expect(dw.has_lo || dx.has_lo, 0)) {
        if (dx.has_lo) {
            const float p_hilo = fdot2_f32_f16(dw.hi, dx.lo, 0.0f);
            acc += scale_pow2_f32(p_hilo, -(dw.shift_hi + dx.shift_lo));
        }
        if (dw.has_lo) {
            const float p_lohi = fdot2_f32_f16(dw.lo, dx.hi, 0.0f);
            acc += scale_pow2_f32(p_lohi, -(dw.shift_lo + dx.shift_hi));
        }
        if (dw.has_lo && dx.has_lo) {
            const float p_lolo = fdot2_f32_f16(dw.lo, dx.lo, 0.0f);
            acc += scale_pow2_f32(p_lolo, -(dw.shift_lo + dx.shift_lo));
        }
    }
    return acc;
}

struct TileScaleState {
    int shift_hi;
    int shift_lo;
    int split_threshold_exp;
    bool needs_dual_bucket;
};

__device__ __forceinline__ void dequant_bf16_tile32_to_dual_f16x2(
    const uint16_t* __restrict__ bf16_ptr,
    __half2* __restrict__ out_hi,
    __half2* __restrict__ out_lo,
    TileScaleState& st
) {
    float vals[32];
    int e_max = -1000;
    int e_min =  1000;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const float v = bf16_bits_to_f32(bf16_ptr[i]);
        vals[i] = v;
        bool is_z = false;
        const int e = extract_unbiased_exp_f32(v, is_z);
        if (!is_z) {
            if (e > e_max) e_max = e;
            if (e < e_min) e_min = e;
        }
    }

    if (e_max == -1000) {
        st.shift_hi = 0;
        st.shift_lo = 0;
        st.split_threshold_exp = -1000;
        st.needs_dual_bucket = false;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            out_hi[j] = __floats2half2_rn(vals[2 * j], vals[2 * j + 1]);
        }
        return;
    }

    st.shift_hi = kTargetUnbiasedExpFp16 - e_max;
    st.split_threshold_exp = e_max - kLosslessExpSpanBf16;
    st.needs_dual_bucket = ((e_max - e_min) > kLosslessExpSpanBf16);
    st.shift_lo = 0;

    if (!st.needs_dual_bucket) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            out_hi[j] = __floats2half2_rn(
                scale_pow2_f32(vals[2 * j],     st.shift_hi),
                scale_pow2_f32(vals[2 * j + 1], st.shift_hi)
            );
        }
    } else {
        int e_lo_max = -1000;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            bool is_z = false;
            const int e = extract_unbiased_exp_f32(vals[i], is_z);
            if (!is_z && e < st.split_threshold_exp && e > e_lo_max) {
                e_lo_max = e;
            }
        }
        st.shift_lo = (e_lo_max > -1000) ? (kTargetUnbiasedExpFp16 - e_lo_max) : 0;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float f0 = vals[2 * j];
            const float f1 = vals[2 * j + 1];
            bool z0 = false, z1 = false;
            const int e0 = extract_unbiased_exp_f32(f0, z0);
            const int e1 = extract_unbiased_exp_f32(f1, z1);

            const bool lo0 = (!z0 && e0 < st.split_threshold_exp);
            const bool lo1 = (!z1 && e1 < st.split_threshold_exp);

            out_hi[j] = __floats2half2_rn(
                lo0 ? 0.0f : scale_pow2_f32(f0, st.shift_hi),
                lo1 ? 0.0f : scale_pow2_f32(f1, st.shift_hi)
            );
            out_lo[j] = __floats2half2_rn(
                lo0 ? scale_pow2_f32(f0, st.shift_lo) : 0.0f,
                lo1 ? scale_pow2_f32(f1, st.shift_lo) : 0.0f
            );
        }
    }
}

} // namespace strata::kernels::gfx906
