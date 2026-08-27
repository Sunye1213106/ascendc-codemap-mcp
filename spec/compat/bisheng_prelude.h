// Minimal Bisheng frontend declarations for vanilla-clang AST recovery.
// Every identifier here is a compiler builtin absent from CANN headers.
// Intrinsics use C varargs (`...`) to match cce_aicore_intrinsics.h aliases —
// call-site arity varies and fixed prototypes produce "no matching function".
#pragma once
#include <cstdint>
#include <cstddef>

// Catlass (template_linear_algebra) writes bisheng postfix attributes
// `__forceinline__[aicore]`. Vanilla clang cannot parse that. Map to the
// same qualifiers erase_qualifiers already empties. Close the include guard
// so operator 3rd copies of macros.hpp do not reintroduce `[aicore]`.
// Operators that `#define HOST_DEVICE __forceinline__ [host, aicore]` in
// their own headers after this prelude still win; those are rewritten via
// ``uo_init.bisheng_attrs`` unsaved_files before libclang parse.
#ifndef CATLASS_DETAIL_MACROS_HPP
#define CATLASS_DETAIL_MACROS_HPP
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#define CATLASS_DEVICE __forceinline__ __aicore__
#define CATLASS_HOST_DEVICE __forceinline__ __host_aicore__
#define CATLASS_GLOBAL __global__ __aicore__
#endif

// From bisheng ``__clang_cce_defines.h``. Vanilla clang has no cce_callee /
// simt attributes; empty the qualifiers so CANN SIMT headers parse.
#ifndef __callee__
#define __callee__
#endif
#ifndef __simt_callee__
#define __simt_callee__
#endif
#ifndef __simd_callee__
#define __simd_callee__
#endif
#ifndef __simt_vf__
#define __simt_vf__
#endif
#ifndef __simd_vf__
#define __simd_vf__
#endif
#ifndef LAUNCH_BOUND
#define LAUNCH_BOUND(...)
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

// ---- scalar builtin types -------------------------------------------------
// Each FP8/FP4 spelling must be a DISTINCT type: CANN specializes templates on
// float8_e4m3_t / float8_e5m2_t / ... separately, and aliasing them all to one
// struct makes those specializations collide as redefinitions.
struct __bs_f16 {
    uint16_t v;
    constexpr __bs_f16() : v(0) {}
    explicit constexpr __bs_f16(int x) : v(static_cast<uint16_t>(x)) {}
    explicit constexpr __bs_f16(unsigned x) : v(static_cast<uint16_t>(x)) {}
    // Bisheng half converts from float. Do not cast the magnitude to
    // uint16_t: ``static constexpr half MIN_VALUE = -65504.0f`` is a
    // constexpr init and the out-of-range conversion is not a constant
    // expression under vanilla clang.
    constexpr __bs_f16(float) : v(0) {}
    explicit constexpr __bs_f16(double) : v(0) {}
    explicit constexpr operator float() const { return float(v); }
};
struct __bs_b16 { uint16_t v; };
struct __bs_f8_e4m3 { uint8_t v; };
struct __bs_f8_e5m2 { uint8_t v; };
struct __bs_f8_e8m0 { uint8_t v; };
struct __bs_hif8 { uint8_t v; };
struct __bs_f4_e2m1x2 { uint8_t v; };
struct __bs_f4_e1m2x2 { uint8_t v; };

using half = __bs_f16;
using float32_t = float;
using bfloat16_t = __bs_b16;
using hifloat8_t = __bs_hif8;
using float8_e4m3_t = __bs_f8_e4m3;
using float8_e5m2_t = __bs_f8_e5m2;
using float8_e8m0_t = __bs_f8_e8m0;
using float4_e2m1x2_t = __bs_f4_e2m1x2;
using float4_e1m2x2_t = __bs_f4_e1m2x2;
// Do not alias fp4x2_* / fp8_* here. tikcfw common_types.h defines those
// as float4_*/float8_* on arch35 and as uint8_t on older arches; a prelude
// using would make the uint8_t branch a type-alias redefinition.
struct __bs_i4x2 { uint8_t v; };
using int4x2_t = __bs_i4x2;

#ifndef uint
using uint = unsigned int;
#endif

// ---- pipe / event / memory handles ---------------------------------------
enum pipe_t {
    PIPE_S = 0, PIPE_V = 1, PIPE_M = 2, PIPE_MTE1 = 3,
    PIPE_MTE2 = 4, PIPE_MTE3 = 5, PIPE_ALL = 6, PIPE_FIX = 7,
    PIPE_MTE4 = 8, PIPE_MTE5 = 9, PIPE_V2 = 10,
};
enum event_t {
    EVENT_ID0 = 0, EVENT_ID1 = 1, EVENT_ID2 = 2, EVENT_ID3 = 3,
    EVENT_ID4 = 4, EVENT_ID5 = 5, EVENT_ID6 = 6, EVENT_ID7 = 7,
};
enum mem_t {
    MEM_UB = 0, MEM_L1 = 1, MEM_L0A = 2, MEM_L0B = 3, MEM_L0C = 4, MEM_GM = 5,
};

// Enums from cce_aicore_intrinsics.h. Stubbed here because that header is only
// reachable after several other builtins parse cleanly; without these stubs
// DMA / pad structs fail early with "unknown type name".
enum QuantMode_t {
    NoQuant = 0,
    F322F16 = 1,
    VQF322HIF8_PRE = 2,
    QF322HIF8_PRE = 3,
    VQF322HIF8_PRE_HYBRID = 4,
    QF322HIF8_PRE_HYBRID = 5,
    AttachF16Mul = 6,
    VDEQS32_INT = 7,
    VREQ8 = 8,
    REQ8 = 9,
    VDEQF16 = 10,
    DEQF16 = 11,
    VQF322FP8_PRE = 12,
    QF322FP8_PRE = 13,
    VQF322F32_PRE = 14,
    QF322F32_PRE = 15,
    F322BF16 = 16,
    VQF162B8_PRE = 17,
    QF162B8_PRE = 18,
    VQF162S4_PRE = 19,
    QF162S4_PRE = 20,
    VREQ4 = 21,
    REQ4 = 22,
    VQF322B8_PRE = 23,
    QF322B8_PRE = 24,
    VQF322S4_PRE = 25,
    QF322S4_PRE = 26,
    VDEQS16 = 27,
    DEQS16 = 28,
    VQF162S16_PRE = 29,
    QF162S16_PRE = 30,
    VQF322F16_PRE = 31,
    QF322F16_PRE = 32,
    VQF322BF16_PRE = 33,
    QF322BF16_PRE = 34,
    VQS322BF16_PRE = 35,
    QS322BF16_PRE = 36,
    DEQS32_INT = 37,
    VSHIFTS322S16 = 38,
    SHIFTS322S16 = 39,
    VREQ16 = 40,
    REQ16 = 41,
};
enum pad_t { PAD_NONE = 0 };
enum cache_line_t { SINGLE_CACHE_LINE = 0, ENTIRE_DATA_CACHE = 1 };
enum dcci_dst_t { CACHELINE_ALL = 0, CACHELINE_OUT = 2 };
enum mem_dsb_t { MEM_DSB_NONE = 0 };
enum Spr { SPR_NONE = 0 };

// cce_aicore_intrinsics.h / __clang_cce_vector_intrinsics.h are compiler
// builtins. CANN headers use these names without including those files.
enum atomic_op_t { ATOMIC_SUM = 0 };
enum atomic_type_t {
    ATOMIC_NONE = 0,
    ATOMIC_F32 = 1,
    ATOMIC_F16 = 2,
    ATOMIC_S16 = 3,
    ATOMIC_S32 = 4,
    ATOMIC_S8 = 5,
    ATOMIC_BF16 = 6,
};
enum class Mode {
    UNKNOWN_VALUE,
    MERGING_VALUE,
    ZEROING_VALUE,
    MERGING_SRC0_VALUE
};

// Bisheng SIMT launch builtins. CANN `kernel_simt_utils.h` writes
// `using Dim3 = cce::dim3` without including a compiler header vanilla clang
// can see. Cube/softmax tiling PODs and RegTensor stay in CANN headers.
namespace cce {
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};
}
using Dim3 = cce::dim3;
struct half2 { half x; half y; };
struct float2 { float x; float y; };

// ---- vector / mask register builtins (Bisheng MicroAPI foundation) -------
// CANN: MaskReg=vector_bool, UnalignRegForStore=vector_align, AddrReg=vector_address.
// Do not stub RegTensor / VecReg here: CANN headers already declare them.
struct vector_bool { uint64_t bits[4]; };
struct vector_align { uint8_t data[32]; };
struct vector_address { uint32_t addr; };
struct vector_u8 { uint8_t v[256]; };
struct vector_u16 { uint16_t v[128]; };
struct vector_u32 { uint32_t v[64]; };
struct vector_u64 { uint64_t v[32]; };
struct vector_s8 { int8_t v[256]; };
struct vector_s16 { int16_t v[128]; };
struct vector_s32 { int32_t v[64]; };
struct vector_s64 { int64_t v[32]; };
struct vector_f16 { uint16_t v[128]; };
struct vector_f32 { float v[64]; };
struct vector_f64 { double v[32]; };
struct vector_bf16 { uint16_t v[128]; };
struct vector_hif8 { uint8_t v[256]; };
struct vector_f8e4m3 { uint8_t v[256]; };
struct vector_f8e5m2 { uint8_t v[256]; };
struct vector_f8e8m0 { uint8_t v[256]; };
struct vector_f4e2m1x2 { uint8_t v[256]; };
struct vector_f4e1m2x2 { uint8_t v[256]; };
// Packed 4-bit lanes (s4x2 / u4x2). Absent from CANN headers under vanilla clang;
// operator TUs then fail probe as `unknown type name 'vector_s4x2'`.
struct vector_s4x2 { uint8_t v[128]; };
struct vector_u4x2 { uint8_t v[128]; };
struct vector_s4 { uint8_t v[128]; };
struct vector_u4 { uint8_t v[128]; };

// Rounding mode tag. Bisheng builtin is `enum class ROUND { R, A, F, C, Z, O, H }`
// in `__clang_dpp_types.h`. An enumerator named ROUND is the wrong stub:
// kernel/CANN code writes `ROUND::…` and then reports
// `'ROUND' is not a class, namespace, or enumeration`.
enum class ROUND { R, A, F, C, Z, O, H };

using MaskReg = vector_bool;
using UnalignRegForLoad = vector_align;
using UnalignRegForStore = vector_align;
using UnalignReg = vector_align;
using AddrReg = vector_address;

namespace __cce_scalar {
inline uint64_t get_ctrl(...) { return 0; }
inline void set_ctrl(...) {}
inline uint64_t sbitset0(...) { return 0; }
inline uint64_t sbitset1(...) { return 0; }
inline void copy_ubuf_to_gm_align_v2(...) {}
inline void copy_ubuf_to_ubuf(...) {}
inline void dcci(...) {}
}

// ---- misc builtin intrinsics (varargs, matching bisheng alias headers) ----
extern "C" {
uint64_t get_ctrl(...);
void set_ctrl(...);
int64_t get_block_idx(...);
int64_t get_block_num(...);
int64_t get_subblockid(...);
int64_t get_subblockdim(...);
int64_t get_coreid(...);
uint64_t get_arch_ver(...);
uint64_t get_pc(...);
uint64_t get_imm(...);
uint64_t get_ar(...);
uint64_t get_rsvd_cnt(...);
uint64_t sbitset0(...);
uint64_t sbitset1(...);
void set_vector_mask(...);
void set_vector_mask_dup(...);
void set_mask_norm(...);
void set_mask_count(...);
void set_atomic_none(...);
void set_padding(...);
void set_loop_size_ubtoout(...);
void set_loop_size_outtoub(...);
void set_st_atomic_cfg(...);
void set_aipp_spr_9(...);
void set_aipp_spr_18(...);
void set_aipp_spr_19(...);
void set_aipp_spr_20(...);
void set_aipp_spr_21(...);
void pipe_barrier(...);
void set_flag(...);
void wait_flag(...);
void hset_flag(...);
void hwait_flag(...);
void dcci(...);
void dsb(...);
void get_buf(...);
void rls_buf(...);
void copy_ubuf_to_gm_align_v2(...);
void copy_ubuf_to_ubuf(...);
void sprclr(...);
int64_t sff0(...);
void trap(...);
}
