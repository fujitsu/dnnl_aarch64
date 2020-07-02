/*******************************************************************************
* Copyright 2019-2020 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#define DEBUG_JIT_UNI_REORDER 0
#define SCALE 0

#define ld1_test(...) ld1(__FILE__,__LINE__,__VA_ARGS__)

#include <assert.h>
#if DEBUG_JIT_UNI_REORDER
#include <iostream>
#define DBG_MSG_JIT_REORDER(str, x) std::cout << __FILE__ << ":" << __LINE__ << ", JIT_REORDER:" << #str<< "=" << x << std::endl;
#else
#define DBG_MSG_JIT_REORDER(str, x)
#endif

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "mkldnn_debug.h"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "cpu_primitive.hpp"
#include "cpu_reorder_pd.hpp"
#include "jit_uni_reorder.hpp"

#include "jit_generator_aarch64.hpp"

#if DEBUG_JIT_UNI_REORDER
#define DEBUG0(x) mov(reg_debug0, x);
#define DEBUG1(x) mov(reg_debug1, x);
#define DEBUG2(x) mov(reg_debug2, x);
#else
#define DEBUG0(x)
#define DEBUG1(x)
#define DEBUG2(x)
#endif

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) do { __VA_ARGS__ } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
#endif

using namespace Xbyak::Xbyak_aarch64;
using namespace mkldnn::impl::types;

namespace mkldnn {
namespace impl {
namespace cpu {

namespace tr {

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32: public kernel_t, public jit_generator_aarch64 {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    enum {
        /* Unrolling in inner dimension order
         * until the number of elements reaches len_unroll_max */
        len_unroll_max = 256, 
        /* Check if it can unroll in order from the innermost dimension,
         * and a jit-function is not generated 
         * if the number of dimensions that could not be unrolled 
         * is larger than or equal to ndim_jit_loop_max. */
        ndims_jit_loop_max
        = 3, 
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll; // Number of full unrolled dimenstions
        int len_last_dim_unroll; 
        int len_unroll;
    };

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int len_unroll = 1; // The total number of elements which is unrolled

        for (int d = 0; d < ndims; ++d) { // for each dimension
            auto &node = prb.nodes[d];
            if (len_unroll * node.n
                    <= len_unroll_max) { // Check if all dimensions can be unrolled
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else { 
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n
                        % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max)
            return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = true && p.ndims > 0
                && utils::one_of(p.itype, f32 /*, s32, s8, u8*/)
                && utils::one_of(p.otype, f32 /*, s32, s8, u8*/)
                && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
                && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
                && simple_impl_desc_init(p, nullptr) && mayiuse(sve);
        if (!ok)
            return false;

        const ptrdiff_t max_stride = (1LL << 31) - 1;
        for (int d = 0; d < p.ndims; ++d) {
            const ptrdiff_t cms = max_stride / p.nodes[d].n;
            bool strides_ok = true
                    && p.nodes[d].is < cms / (int)data_type_size(p.itype)
                    && p.nodes[d].os < cms / (int)data_type_size(p.otype);
            if (!strides_ok)
                return false;
        }

        return true;
    }

    int n(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].n;
    }
    int is(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].is;
    }
    int os(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].os;
    }
    int ss(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].ss;
    }

    AdrScImm i_addr(const ZRegS &v, int i_off, int simd_w) {
        // reg_ptr_in : Base address of input array.
        // reg_off_in : Num of offset data.
        // i_off :      Temporal num of offset data.
        // Current position = reg_ptr_in + (reg_off_in + i_off) * itype_sz
        assert(!(i_off % simd_w));
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_in, i_off / simd_w, MUL_VL);
    }

    AdrScImm i_addr(const VReg4S &v, int i_off, int simd_w) {
        assert(!(i_off % simd_w));
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_in, i_off / simd_w, MUL_VL);
    }

    AdrImm i_addr(int i_off) {
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_in, i_off * itype_sz);
    }

    AdrScImm o_addr(const ZRegS &v, int o_off, int simd_w) {
        assert(!(o_off % simd_w));
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_out, o_off / simd_w, MUL_VL);
    }

    AdrScImm o_addr(const VReg4S &v, int o_off, int simd_w) {
        assert(!(o_off % simd_w));
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_out, o_off / simd_w, MUL_VL);
    }

    AdrImm o_addr(int o_off) {
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_out, o_off * otype_sz);
    }

    AdrScImm s_addr(const ZRegS &v, int s_off, int simd_w) {
        assert(!(s_off % simd_w));
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_out, s_off / simd_w, MUL_VL);
    }

    AdrScImm s_addr(const VReg4S &v, int s_off, int simd_w) {
        assert(!(s_off % simd_w));
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_out, s_off / simd_w, MUL_VL);
    }

    AdrImm s_addr(int s_off) {
        return Xbyak::Xbyak_aarch64::ptr(reg_ptr_scale, s_off * stype_sz);
    }
    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int &i_off, int &o_off, int &s_off, int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;

        if (off == 0)
            return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int d = start_dim; d < prb_.ndims; ++d) {
            i_off += is(d);
            o_off += os(d);
            s_off += ss(d);

            if (off % n(d))
                break;

            i_off += -n(d) * is(d);
            o_off += -n(d) * os(d);
            s_off += -n(d) * ss(d);
            off /= n(d);

            if (off == 0)
                break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, i_off, o_off, dummy,
                step_size);
    }

    void tr8x8_sve(int i_off, int o_off) {
        using namespace data_type;

        // Load 512bits(32bits x 16words) = 16x4 words = 64 words.
        ld4w(z0.s, p0 / Xbyak::Xbyak_aarch64::T_z, ptr(reg_ptr_in, reg_off_in, LSL, 2));

        // Convert FP <-> Int, if needed.
        for (int i = 0; i < 4; i++) {
            if (prb_.itype == s32 && prb_.otype == f32) {
                fcvtzs(ZReg(i).s, p0 / Xbyak::Xbyak_aarch64::T_m, ZReg(i).s);
            } else if (prb_.itype == f32 && prb_.otype == s32) {
                fcvtzs(ZReg(i).s, p0 / Xbyak::Xbyak_aarch64::T_m, ZReg(i).s);
            }
        }

        uzp1(z4.s, z0.s, z1.s);
        uzp1(z5.s, z2.s, z3.s);
        uzp2(z6.s, z0.s, z1.s);
        uzp2(z7.s, z2.s, z3.s);

        // Store 512bits(32bits x 16words) = 16x4 words = 64 words.
        st4w(z4.s, p0 / Xbyak::Xbyak_aarch64::T_z, ptr(reg_ptr_out, reg_off_out, LSL, 2));
    }

    bool process_unroll_tr8x8(int len) {
        bool can_do = true && mayiuse(avx2) && prb_.ndims >= 2
                && utils::everyone_is(4, itype_sz, otype_sz)
                && utils::everyone_is(8, n(0), n(1))
                && utils::everyone_is(1, os(0), is(1))
                && utils::everyone_is(8, os(1), is(0))
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
        if (!can_do)
            return false;

        const int step_size = n(0) * n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_sve(i_off, o_off);
        }

        return true;
    }

    bool process_direct_copy_sve(int len) {
      using namespace data_type;

      const int simd_w = cpu_isa_traits<sve>::vlen / itype_sz;
      bool isSameType = prb_.itype == prb_.otype ? true : false;

      bool can_do = true && mayiuse(sve)
                         && utils::everyone_is(1, os(0), is(0))
                         && (false || isSameType
                            || (prb_.itype == s32 && prb_.otype == f32))
                         && len % simd_w == 0 && n(0) % len == 0
                         && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
      if (!can_do)
        return false;

      ptrue(reg_p_all_one.b); // Set all bits to 1.

      for (int off = 0; off < len;) {
        int ur;

        // Max # of unroll data is decided by #
        // of SVE registers and their width.
        const int unroll = nstl::min(32, (len - off) / simd_w);

        ur = 0;

        while ((unroll - ur >= 4) && (ur + 4 <= cpu_isa_traits<sve>::n_vregs)) {
          add_imm(reg_tmpIn, reg_ptr_in, (off+ur*simd_w) * itype_sz, reg_tmp);
          add(reg_tmpIn, reg_tmpIn, reg_off_in);
          rsvdOffsetIn = off * itype_sz;
          ld4w(ZReg(ur).s, reg_p_all_one.s, ptr(reg_tmpIn));
          ur += 4;
        }

        while (((unroll - ur) >= 1) && ((ur + 1) <= cpu_isa_traits<sve>::n_vregs)) {
          add_imm(reg_tmpIn, reg_ptr_in, (off+ur*simd_w)* itype_sz, reg_tmp);
          add(reg_tmpIn, reg_tmpIn, reg_off_in);
          rsvdOffsetIn = off * itype_sz;

          ld1w(ZReg(ur).s, reg_p_all_one.s, ptr(reg_tmpIn));

          ur += 1;
        }
        if (prb_.itype != prb_.otype) {
          for (ur = 0; ur < unroll; ++ur) {
            ZRegS zs(ur);
            if (prb_.itype == s32 && prb_.otype == f32)
              scvtf(zs, reg_p_all_one.s, zs);
            else if (prb_.itype == f32 && prb_.otype == s32)
              fcvtzs(zs, reg_p_all_one.s, zs);
            else
              assert(!"unreachable");
          }
        }

        ur = 0;

        while ((unroll - ur >= 4) && (ur + 4 <= cpu_isa_traits<sve>::n_vregs)) {
          add_imm(reg_tmpOut, reg_ptr_out, (off+ur*simd_w) * otype_sz, reg_tmp);
          add(reg_tmpOut, reg_tmpOut, reg_off_out);
          rsvdOffsetOut = (off+ur*simd_w) * otype_sz;

          st4w(ZReg(ur).s, reg_p_all_one.s, ptr(reg_tmpOut));
          ur += 4;
        }

        while (((unroll - ur) >= 1) && ((ur + 1) <= cpu_isa_traits<sve>::n_vregs)) {
          add_imm(reg_tmpOut, reg_ptr_out, (off+ur*simd_w) * otype_sz, reg_tmp);
          add(reg_tmpOut, reg_tmpOut, reg_off_out);
          rsvdOffsetOut = (off+ur*simd_w) * otype_sz;
          st1w(ZReg(ur).s, reg_p_all_one.s, ptr(reg_tmpOut));
          ur += 1;
        }
        off += unroll * simd_w;
      }
      return true;
    }

    bool process_direct_copy_simd(int len) {
      using namespace data_type;

      const int simd_w = cpu_isa_traits<simd>::vlen / itype_sz;
      bool isSameType = prb_.itype == prb_.otype ? true : false;

      bool can_do = true && mayiuse(simd)
                    && utils::everyone_is(1, os(0), is(0))
                    && (false || isSameType
                      || (prb_.itype == s32 && prb_.otype == f32)
                      || (prb_.itype == f32 && prb_.otype == s32))
                    && len % simd_w == 0 && n(0) % len == 0
                    && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;

      if (!can_do)
        return false;

      ptrue(reg_p_all_one.b); // Set all bits to 1.

      for (int off = 0; off < len;) {
        int ur;
        // Max # of unroll data is decided by #
        // of SVE registers and their width.
        const int unroll = nstl::min(32,
                          (len - off) / simd_w);

        if(rsvdOffsetIn != off * itype_sz) {
          add_imm(reg_tmpIn, reg_ptr_in, off * itype_sz, reg_tmp);
          rsvdOffsetIn = off * itype_sz;
        }
        ur = 0;
        while ((unroll - ur >= 4) && (ur + 4 <= cpu_isa_traits<simd>::n_vregs)) {
          ld4((VReg(ur).s4 - VReg(ur + 3).s4), post_ptr(reg_tmpIn, 64));
          rsvdOffsetIn += 64;
          ur += 4;
        }

        // Residual
        if (unroll - ur >= 3) {
          ld3((VReg(ur).s4 - VReg(ur + 2).s4), post_ptr(reg_tmpIn, 48));
          rsvdOffsetIn += 48;
          ur += 3;
        }else if (unroll - ur >= 2) {
          ld2((VReg(ur).s4 - VReg(ur + 1).s4), post_ptr(reg_tmpIn, 32));
          rsvdOffsetIn += 32;
          ur += 2;
        }else if (unroll - ur >= 1) {
          ld1(VReg(ur).s, post_ptr(reg_tmpIn, 16));
          rsvdOffsetIn += 16;
          ur += 1;
        }

        if (prb_.itype != prb_.otype) {
          for (int ur = 0; ur < unroll; ++ur) {
            if (prb_.itype == s32 && prb_.otype == f32)
              scvtf(VReg(ur).s4, VReg(ur).s4);
            else if (prb_.itype == f32 && prb_.otype == s32)
              fcvtas(VReg(ur).s4, VReg(ur).s4);
            else
              assert(!"unreachable");
          }
        }

        if(rsvdOffsetOut != off * otype_sz) {
          add_imm(reg_tmpOut, reg_ptr_out, off * otype_sz, reg_tmp);
          rsvdOffsetOut = off * otype_sz;
        }
        ur = 0;
        while ((unroll - ur >= 4) && (ur + 4 <= cpu_isa_traits<simd>::n_vregs)) {
          st4((VReg(ur).s4 - VReg(ur + 3).s4), post_ptr(reg_tmpOut, 64));
          rsvdOffsetOut += 64;
          ur += 4;
        }

        // Residual
        if (unroll - ur >= 3) {
          st3((VReg(ur).s4 - VReg(ur + 2).s4), post_ptr(reg_tmpOut, 48));
          rsvdOffsetOut += 48;
          ur += 3;
        }else if (unroll - ur >= 2) {
          st2((VReg(ur).s4 - VReg(ur + 1).s4), post_ptr(reg_tmpOut, 32));
          rsvdOffsetOut += 32;
          ur += 2;
        }else if (unroll - ur >= 1) {
          st1(VReg(ur).s, post_ptr(reg_tmpOut, 16));
          rsvdOffsetOut += 16;
          ur += 1;
        }

        off += unroll * simd_w;
      }
      return true;
    }
	
	void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off) {
        using namespace data_type;

        /* Case1: src is a register */
        auto cvt2ps = [=](const VReg &dst, const VReg &src, data_type_t idt) {
            switch (idt) {
            case f32: {
                if (src.getIdx() != dst.getIdx())
                    mov(dst.b16, src.b16);
            } break;
            case s32: scvtf(dst.s4, src.s4); break;
            case data_type::s8:
                /* 8-bit x 4 elements 
                 * -> 16-bit x 4 elem 
                 * -> 32-bit x 4 elem
                 * convert to float */
                sxtl(dst.h8, src.b8);
                sxtl(dst.s4, dst.h4);
                scvtf(dst.s4, dst.s4);
                break;
            case u8:
                /* 8-bit x 4 elements 
                 * -> 16-bit x 4 elem 
                 * -> 32-bit x 4 elem
                 * convert to float */
                uxtl(dst.h8, src.b8);
                uxtl(dst.s4, dst.h4);
                ucvtf(dst.s4, dst.s4);
                break;
            default: assert(!"unreachable");
            }
        };

        /* Case2: src is address op */
        auto cvt2ps_addr = [=](const VReg &dst, const AdrImm &src,
                                   data_type_t idt) {
				
            mov(reg_tmp, XReg(src.getXn()));
            add(reg_tmp, reg_tmp, src.getImm()); //TODO: check
            AdrNoOfs addr(reg_tmp);

            switch (idt) {
            case f32: ld1(dst.s4, addr); break;
            case s32:
                ld1(dst.s4, addr);
                scvtf(dst.s4, dst.s4);
                break;
            case data_type::s8:
                ld1(dst.s4, addr);
                /* 8-bit x 4 elements 
                 * -> 16-bit x 4 elem 
                 * -> 32-bit x 4 elem
                 * convert to float */
                sxtl(dst.h8, dst.b8);
                sxtl(dst.s4, dst.h4); //The bit lenght of lower 4 elements becomes twice
                scvtf(dst.s4, dst.s4);
                break;
            case u8:
                ld1(dst.s4, addr);
                /* 8-bit x 4 elements 
                 * -> 16-bit x 4 elem 
                 * -> 32-bit x 4 elem
                 * convert to float */
                uxtl(dst.h8, dst.b8);
                uxtl(dst.s4, dst.h4); //The bit lenght of lower 4 elements becomes twice
                ucvtf(dst.s4, dst.s4);
                break;
            default: assert(!"unreachable");
            }
        };

        auto cvt2int = [=](const VReg &xmm, data_type_t odt, data_type_t idt) {
            switch (odt) {
            case s32:
                if (idt == f32) { // f32 -> s32
                    fcvtas(xmm.s4, xmm.s4);
                } else if (idt == data_type::s8) { // f32 -> s8
                    sxtl(xmm.h8, xmm.b8); // signed 8-bit -> signed 16-bit
                    sxtl(xmm.s4, xmm.h4); // signed 16-bit -> signed 32-bit
                    scvtf(xmm.s4, xmm.s4); // s32 -> f32
                } else if (idt == u8) { // f32 -> u8
                    uxtl(xmm.h8, xmm.b8); // unsigned 8-bit -> unsigned 16-bit
                    uxtl(xmm.s4, xmm.h4); // unsigned 16-bit -> unsigned 32-bit
                    ucvtf(xmm.s4, xmm.s4); // unsigned 32-bit -> f32
                }
                break;
            case data_type::s8:
                if (idt == f32) { // f32 -> s8
                    fcvtas(xmm.s4, xmm.s4); // f32 -> signed 32-bit
                }
                if (idt == f32 || idt == s32) { // signed 32-bit -> signed 8-bit
                    sqxtn(xmm.h4, xmm.s4); // signed 32-bit -> signed 16-bit
                    sqxtn(xmm.b8, xmm.h8); // signed 16-bit -> signed 8-bit
                }
                if (idt == u8) { // u8 -> s8
                    /* sqadd : signed saturating add
                     * Input is u8, and output is s8.
                     * If input is larger than 127, the value should be round to 127.*/
                    sqadd(xmm.b16, xmm.b16, xmm_zero.b16);
                }
                break;
            case u8:
                if (idt == f32) { // f32 -> u8
                    fcvtau(xmm.s4, xmm.s4); // f32 -> unsigned 32-bit
                }
                if (idt == f32
                        || idt == s32) { // unsigned 32-bit -> unsigned 8-bit
                    uqxtn(xmm.h4, xmm.s4); // unsigned 32-bit -> unsigned 16-bit
                    uqxtn(xmm.b8, xmm.h8); // unsigned 16-bit -> unsigned 8-bit
                }
                if (idt == data_type::s8) { // s8 -> u8
                    /* Input is s8, and output is u8.
                     * If input is a negative value, the value should be round to 0.
                     * Op1 is signed, and Op2 is unsigned. */
                    usqadd(xmm.b16, xmm_zero.b16);
                }
                break;
            default: assert(!"unreachable");
            }
        };

        auto loadPost = [=](const AdrPostImm &addr, const VReg &xmm, int size) {
            /* Warning: there is a possibility that addr is not aligned. */
            switch (size) {
            case 16:
                ld1(xmm.s4, addr); // load 128 bits
                break;
            case 4:
                ld1((xmm.s)[0], addr); // load 32 bits
                break;
            case 1: 
                ld1((xmm.b)[0], addr); // load 8 bits (LSB). Other bits are
                                      // 0 cleared.
                break;
            default: assert(!"unreachable");
            }
        };

        auto storePost = [=](const AdrPostImm &addr, const VReg &xmm, int size) {
            switch (size) {
            case 16:
                st1(xmm.s4, addr); // load 128 bits
                break;
            case 4:
                st1((xmm.s)[0], addr);
                break;
            case 1: 
                st1((xmm.b)[0], addr);
                break;
            default: assert(!"unreachable");
            }
        };

        /* Check whether loading 4 values at once is possible */
        bool can_load_xmm = mayiuse(simd) && reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1)
                can_load_xmm = false;
        const int load_step = can_load_xmm ? 4 : 1;

        /* Check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1)
                can_store_xmm = false;
        const int ur_step = can_store_xmm ? 4 : 1;

        DBG_MSG_JIT_REORDER(ur_step, ur_step);

        const bool interim_f32 = false
                || utils::one_of(f32, prb_.itype, prb_.otype)
                || prb_.scale_type != scale_type_t::NONE || prb_.beta != 0.f;

        if (!can_load_xmm && can_store_xmm) {
            assert(ur_step == 4);
            /* load with stride */
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                for (int r = 0; r < ur_step; ++r) {
                    int imm = i_off[ur + r] * itype_sz;

                    DBG_MSG_JIT_REORDER(load with stride addr, imm);

                    if(rsvdOffsetIn != imm) {
                        add_imm(reg_tmpIn, reg_ptr_in, imm, reg_tmp);
                        rsvdOffsetIn = imm;
                    }

                    AdrPostImm addr(reg_tmpIn, itype_sz);

                    if (itype_sz == 4) {
                        ld1((VReg(ur).s4)[r], addr);
                    } else {
                        ld1((VReg(ur).b16)[r], addr);
                    }
                    rsvdOffsetIn += itype_sz;
                }
            }
        } else {
            if(i_off[0] * itype_sz != rsvdOffsetIn) {
                add_imm(reg_tmpIn, reg_ptr_in, 
                        i_off[0] * itype_sz, reg_tmp); // Set 0-th addresss
                rsvdOffsetIn = i_off[0] * itype_sz;
            }

            for (int ur = 0; ur < reg_unroll; ur += load_step) {
                if(rsvdOffsetIn != i_off[ur] * itype_sz) { // Address is not preset.
                    add_imm(reg_tmpIn, reg_ptr_in, 
                            i_off[ur] * itype_sz, reg_tmp);
                    rsvdOffsetIn = i_off[ur] * itype_sz;
                }
	    
                loadPost(AdrPostImm(reg_tmpIn, load_step*itype_sz), 
                                    VReg(ur), load_step*itype_sz);
                rsvdOffsetIn += load_step * itype_sz;
            }
        }
	
        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(VReg(ur), VReg(ur), prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            const bool fast_return = true // transposition on the fly
                    && prb_.scale_type != scale_type_t::MANY
                    && prb_.beta == 0.f;
            if (fast_return) {
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    if (prb_.scale_type == scale_type_t::COMMON)
                        fmul(VReg(ur).s4, VReg(ur).s4, xmm_scale.s4);
                    if (prb_.otype != f32)
                        cvt2int(VReg(ur), prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                    for (int r = 0; r < load_step; ++r) {
                        add_imm(reg_tmpOut, reg_ptr_out, 
                                o_off[ur + r] * otype_sz, reg_tmp);

                        DBG_MSG_JIT_REORDER(store, o_off[ur + r] * otype_sz);

                        AdrPostImm addr(reg_tmpOut, otype_sz);
                        if (otype_sz == 4) {
                            st1((VReg(ur).s4)[r], addr);
                        } else {
                            assert(load_step == 1);
                            st1((VReg(ur).b8)[r], addr);
                        }
                        rsvdOffsetOut += otype_sz;
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        // vshufps(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
                        ins_((VReg(ur + r).s4)[0], (VReg(ur).s4)[r % 4]);
                        ins_((VReg(ur + r).s4)[1], (VReg(ur).s4)[(r >> 2) % 4]);
                        ins_((VReg(ur + r).s4)[2], (VReg(ur).s4)[(r >> 4) % 4]);
                        ins_((VReg(ur + r).s4)[3], (VReg(ur).s4)[(r >> 6) % 4]);
                    }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step;
                            ++r) // concatenate two vectors and shift right by r
                                 // elements.
                        ext(VReg(ur + r).b16, VReg(ur).b16, VReg(ur).b16, r);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            /* xmm <-- scale * xmm[:] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    fmul(VReg(ur).s4, VReg(ur).s4, xmm_scale.s4);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                enum class scale_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    scale_load_type_t scale_load_type
                            = scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast) {

                        add_imm(reg_tmpScale, reg_ptr_scale, 
                                s_off[ur] * stype_sz, reg_tmp);
                        add(reg_tmpScale, reg_tmpScale, reg_off_scale);
                        ld1((xmm_scale.s4)[0], ptr(reg_tmpScale));

                        dup(xmm_scale.s4, (xmm_scale.s4)[0]);
                        fmul(VReg(ur).s4, VReg(ur).s4, xmm_scale.s4);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load) {
                        int64_t tmpOffset = s_off[ur] * stype_sz;
                        add_imm(reg_tmpScale, reg_ptr_scale, tmpOffset, reg_tmp);
                        add(reg_tmpScale, reg_tmpScale, reg_off_scale);


                        ld1(xmm_scale.s4, ptr(reg_tmpScale));
                        fmul(VReg(ur).s4, VReg(ur).s4, xmm_scale.s4);
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    for (int r = ur; r < ur + ur_step; ++r) {
                        int64_t tmpOffset = s_off[r] * stype_sz;
                        add_imm(reg_tmpScale, reg_ptr_scale, 
                                tmpOffset, reg_tmp);
                        add(reg_tmpScale, reg_tmpScale, reg_off_scale);
                        ld1((xmm_scale.s4)[r - ur], ptr(reg_tmpScale));
                    }
                    fmul(VReg(ur).s4, VReg(ur).s4, xmm_scale.s4);
                }
            }

            /* dst <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        int64_t tmpOffset = o_off[ur] * otype_sz;
                        if(rsvdOffsetOut != tmpOffset) {
                            add_imm(reg_tmpOut, reg_ptr_out, 
                                    tmpOffset, reg_tmp);
                            rsvdOffsetOut = tmpOffset;
                        }
                        AdrPostImm addr(reg_tmpOut, 4);
                        ld1(xmm_tmp.s4, addr);
                        rsvdOffsetOut += 4;
                        fadd(VReg(ur).s4, VReg(ur).s4, xmm_tmp.s4);
                    } else {
                        cvt2ps_addr(xmm_tmp, o_addr(o_off[ur]), prb_.otype);
                        fadd(VReg(ur).s4, VReg(ur).s4, xmm_tmp.s4);
                    }
                }
            }
        } else {
            /* xmm[0] <-- scale * xmm[0] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    /* DST[31:0] = (LSB 32 bits) * (LSB 32 bits)
                       DST[127:32] = DST[127:32] */
                    fmul(xmm_tmp.s4, VReg(ur).s4, xmm_scale.s4);
                    mov((VReg(ur).s4)[0], (xmm_tmp.s4)[0]);
                }
            } else if (prb_.scale_type == scale_type_t::MANY) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    int64_t tmpOffset = s_off[ur] * stype_sz;
                    if(rsvdOffsetScale != tmpOffset) {
                        add_imm(reg_tmpScale, reg_ptr_scale, tmpOffset, reg_tmp);
                        rsvdOffsetScale = tmpOffset;
                    }
                    AdrPostImm addr(reg_tmpScale, 4);
                    ld1((xmm_tmp.s4)[0], addr);
                    rsvdOffsetScale += 4;
                    /* DST[31:0] = (LSB 32 bits) * (LSB 32 bits)
                       DST[127:32] = DST[127:32] */
                    fmul(xmm_tmp.s4, VReg(ur).s4, (xmm_tmp.s4)[0]);
                    mov((VReg(ur).s4)[0], (xmm_tmp.s4)[0]);
                }
            }

            /* dst <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    int64_t tmpOffset = o_off[ur] * otype_sz;
                    if(rsvdOffsetOut != tmpOffset) {
                        add_imm(reg_tmpOut, reg_ptr_out, tmpOffset, reg_tmp);
                        rsvdOffsetOut = tmpOffset;
                    }

		  
                    if (prb_.otype == f32) {
                        AdrPostImm addr(reg_tmpOut, 4);
                        ld1((xmm_tmp.s4)[0], addr);
                        rsvdOffsetOut += 4;
                        fadd(xmm_tmp.s4, VReg(ur).s4, (xmm_tmp.s4));
                        CodeGeneratorAArch64::mov((VReg(ur).s4)[0], (xmm_tmp.s4)[0]);
                    } else {
                        if (prb_.otype == s32) {
                            AdrPostImm addr(reg_tmpOut, 4);
                            ld1((xmm_tmp.s4)[0], addr);
                            rsvdOffsetOut += 4;
                        } else if (utils::one_of(prb_.otype, s8, u8)) {
                            AdrPostImm addr(reg_tmpOut, 1);
                            ld1((xmm_tmp.b16)[0], addr);
                            rsvdOffsetOut += 1;
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(xmm_tmp, xmm_tmp, prb_.otype);
                        CodeGeneratorAArch64::fadd(VReg(ur).s4, VReg(ur).s4, xmm_tmp.s4);
                    }
                }
            }
        }


        if (prb_.otype != f32) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                cvt2int(VReg(ur), prb_.otype, interim_f32 ? f32 : prb_.itype);
            }
        }

        if(o_off[0] * otype_sz != rsvdOffsetOut) {
            add_imm(reg_tmpOut, reg_ptr_out, 
                    o_off[0] * otype_sz, reg_tmp); // Set 0-th address
            rsvdOffsetOut = o_off[0] * otype_sz;
        }
	
        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if(rsvdOffsetOut != o_off[ur] * otype_sz) { // Address is not preset.
                add_imm(reg_tmpOut, reg_ptr_out, 
                        o_off[ur] * otype_sz, reg_tmp); 
                rsvdOffsetOut = o_off[ur] * otype_sz;
            }

            storePost(AdrPostImm(reg_tmpOut, ur_step*otype_sz), 
                      VReg(ur), ur_step * otype_sz); // Prepare next address
            rsvdOffsetOut += ur_step * otype_sz;
        }
    }

    void process_unroll_generic(int len) {
        const int blk = 8;

        int i_off[2 * blk] = { 0 };
        int o_off[2 * blk] = { 0 };
        int s_off[2 * blk] = { 0 };

        int curr = 0; // will switch between 0 and 1

        for (int off = 0; off < len; off += blk) {
            // reg_unroll = blk or residual
            const int reg_unroll = nstl::min(off + blk, len) - off; 

            /* compute offsets */
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr * blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        i_off[ur_c], o_off[ur_c], s_off[ur_c]);
            }

            process_unroll_generic_step(reg_unroll,
                    i_off + curr * blk, //Alternate use of first and second half of
                                        //i_off, o_off, and s_off
                    o_off + curr * blk, s_off + curr * blk);

            curr = 1 - curr;
        }
    }

    void loop_begin(LabelAArch64 &l, XReg reg_cnt, int len) {
        CodeGeneratorAArch64::mov(reg_cnt, len);
        L_aarch64(l);
    }

    void loop_end(LabelAArch64 &l, XReg reg_cnt, int len, int i_step, int o_step,
            int s_step) {

        bool flag = (prb_.scale_type == scale_type_t::MANY);
        int iTmp = i_step * itype_sz;
        int oTmp = o_step * otype_sz;


        DBG_MSG_JIT_REORDER(add load addr, iTmp);
        DBG_MSG_JIT_REORDER(add store addr, iTmp);

        add_imm(reg_ptr_in, reg_ptr_in, iTmp, reg_tmp);
        add_imm(reg_ptr_out, reg_ptr_out, oTmp, reg_tmp);

        if (flag) {
            add_imm(reg_ptr_scale, reg_ptr_scale, s_step * stype_sz, reg_tmp);
        }
        CodeGeneratorAArch64::sub(reg_cnt, reg_cnt, 1);
        cbnz(reg_cnt, l);


        DBG_MSG_JIT_REORDER(sub load addr, iTmp*len);
        DBG_MSG_JIT_REORDER(sub store addr, iTmp*len);

        sub_imm(reg_ptr_in, reg_ptr_in, iTmp * len, reg_tmp);
        sub_imm(reg_ptr_out, reg_ptr_out, oTmp * len, reg_tmp);

        if (flag) {
            sub_imm(reg_ptr_scale, reg_ptr_scale, len * s_step * stype_sz, reg_tmp);
        }
    }

    bool simple_impl() {
        simple_impl_desc_t d;

        if (!simple_impl_desc_init(prb_, &d))
            return false;

        const int nfu = d.ndims_full_unroll;
        const int ldu = d.len_last_dim_unroll;
        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        assert(n_jit_loops <= ndims_jit_loop_max);

        eor(reg_off_in, reg_off_in, reg_off_in);
        eor(reg_off_out, reg_off_out, reg_off_out);
        if (prb_.scale_type == scale_type_t::MANY)
            eor(reg_off_scale, reg_off_scale, reg_off_scale);

        LabelAArch64 l_loop[3];

        if (n_jit_loops > 2)
            loop_begin(l_loop[2], reg_cnt[2], n(nfu + 2));

        if (n_jit_loops > 1)
            loop_begin(l_loop[1], reg_cnt[1], n(nfu + 1));

        if (n_jit_loops > 0)
            loop_begin(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu);

        const bool optimized = false || process_direct_copy_sve(d.len_unroll)
               ||  process_direct_copy_simd(d.len_unroll);
               //  || process_unroll_tr8x8(d.len_unroll); // under construction
        if (!optimized)
            process_unroll_generic(d.len_unroll);

        if (n_jit_loops > 0)
            loop_end(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu, is(nfu + 0) * ldu,
                    os(nfu + 0) * ldu, ss(nfu + 0) * ldu);

        if (n_jit_loops > 1)
            loop_end(l_loop[1], reg_cnt[1], n(nfu + 1), is(nfu + 1),
                    os(nfu + 1), ss(nfu + 1));

        if (n_jit_loops > 2)
            loop_end(l_loop[2], reg_cnt[2], n(nfu + 2), is(nfu + 2),
                    os(nfu + 2), ss(nfu + 2));

        return true;
    }

    void impl() {
        if (simple_impl())
            return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32(const desc_t &desc)
        : kernel_t(desc), jit_generator_aarch64() {
        itype_sz = data_type_size(prb_.itype); // Number of byte per input data
        otype_sz = data_type_size(prb_.otype); // Number fo byte per output data
        stype_sz = sizeof(float); // Number of byte per scale data

        preamble(); // Base on ABI specification, save calle-saved
                    // registers.
        if (prb_.scale_type == scale_type_t::COMMON) {
            ldr(reg_tmp, Xbyak::Xbyak_aarch64::ptr(abi_param1_aarch64, static_cast<int32_t>(offsetof(call_param_t, scale))));
            ld1( {xmm_scale.s4 }, Xbyak::Xbyak_aarch64::ptr(reg_tmp));
        } else if (prb_.scale_type == scale_type_t::MANY) {
            ldr(reg_ptr_scale, Xbyak::Xbyak_aarch64::ptr(abi_param1_aarch64, static_cast<int32_t>(offsetof(call_param_t, scale))));
        }
        ldr(reg_ptr_in,
            Xbyak::Xbyak_aarch64::ptr(abi_param1_aarch64, static_cast<int32_t>(offsetof(call_param_t, in)))); // Store base address of input data to a
                            // register.
        ldr(reg_ptr_out,
            Xbyak::Xbyak_aarch64::ptr(abi_param1_aarch64, static_cast<int32_t>(offsetof(call_param_t, out)))); // Store base address of output data to a
                             // register.
#undef PARAM

        eor(reg_zero, reg_zero, reg_zero);

        if (mayiuse(sve)) {
            eor(xmm_zero.b16, xmm_zero.b16, xmm_zero.b16); // Zero clear

            if (prb_.itype == data_type::u8
                    && prb_.otype == data_type::s8) { // Generate mask
                movi(xmm_zero.b16, 0x7f);
            }
        }

        impl();

        postamble();

        ready();
        ker_ = (void (*)(const call_param_t *))getCode32();

        rsvdOffsetIn = 0xFFFFFFFFFFFFFFFF;
        rsvdOffsetOut = 0xFFFFFFFFFFFFFFFF;
        rsvdOffsetScale = 0xFFFFFFFFFFFFFFFF;

    }




private:
    int itype_sz;
    int otype_sz;
    int stype_sz;


#if DEBUG_JIT_UNI_REORDER
	int debug0 = 0;
	int debug1 = 0;
	int debug2 = 0;

  XReg reg_debug0 = x21;
  XReg reg_debug1 = x22;
  XReg reg_debug2 = x23;
#endif

    XReg reg_ptr_in = x9;
    XReg reg_ptr_out = x10;
    XReg reg_ptr_scale = x11;

    XReg reg_off_in = x12;
    XReg reg_off_out = x13;
    XReg reg_off_scale = x14;

    XReg reg_tmp = x15;
  XReg reg_tmpIn = x17;
  XReg reg_tmpOut = x18;
  XReg reg_tmpScale = x19;

  XReg reg_zero = x20;
  XReg reg_cnt[3] = { x28, x27, x26 };

  
    VReg xmm_scale = v15;
    VReg xmm_zero = v14;
    VReg xmm_4x127b = v13; // TODO: unite with xmm_zero
    VReg xmm_tmp = v12;

    PReg reg_p_all_zero = p0;
    PReg reg_p_all_one  = p1;

  int64_t rsvdOffsetIn = 0xFFFFFFFFFFFFFFFF;
  int64_t rsvdOffsetOut = 0xFFFFFFFFFFFFFFFF;
  int64_t rsvdOffsetScale = 0xFFFFFFFFFFFFFFFF;
}; // namespace tr

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims)
        return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min)
                return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0)
        ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
    case 0: return new jit_uni_reorder_kernel_f32(desc);
    default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

} // namespace tr

static void prb_block_for_cache(tr::prb_t &prb) {
    if (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16) {
        /** an attempt to use caches more efficient and
         * address the 4K-aliasing issue */
        /* TODO: improve the logic around here */
        int j = 1;
        for (; j < prb.ndims && prb.nodes[j].is != 1; ++j)
            ;
        if (j == prb.ndims)
            return;

        /* it makes sense to re-prioritize sequential read over
         * sequential write if the former would not trash the
         * cache, i.e. is == 1 and os % 2^smth != 0. Smth is
         * set to 2 at the moment */
        const int move_to = prb.nodes[j].os % 4 != 0 ? 0 : 1;
        if (j == move_to)
            return;

        if (prb.nodes[j].n > 16 && prb.nodes[j].n % 16 == 0)
            prb_node_split(prb, j, 16);

        prb_node_move(prb, j, move_to);
        DEBUG({
            printf("cache: ");
            prb_dump(prb);
        });
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(tr::prb_t &prb, int &ndims_ker_max) {
    size_t sz_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        sz_total *= prb.nodes[d].n;

    /* sz_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t sz_drv_min = nstl::min<size_t>(
            16 * mkldnn_get_max_threads(), utils::div_up(sz_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * sz_ker_cur -- product of the dimension processed by a kernel
     * sz_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t sz_drv_cur = 1;
    for (; kdims > 1 && sz_drv_cur < sz_drv_min; --kdims)
        sz_drv_cur *= prb.nodes[kdims - 1].n;

    size_t sz_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        sz_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that sz_drv_cur >= sz_drv_min.
     *
     * It might happen that for chosen kdims the sz_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase sz_ker_cur. */
    bool want_borrow_ker_from_drv = true && kdims < prb.ndims
            && sz_ker_cur < tr::ker_prb_size_min && sz_drv_cur > sz_drv_min;
    if (want_borrow_ker_from_drv) {
        /* sz_want_borrow is the minimal sz, so that:
         *  o) sz_ker_cur * sz_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     sz_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal sz_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t sz_want_borrow = utils::div_up(tr::ker_prb_size_min, sz_ker_cur);
        for (; prb.nodes[kdims].n % sz_want_borrow; ++sz_want_borrow)
            ;
        if (sz_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, sz_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the sz_drv_cur is too small (less than sz_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * sz_drv_cur. */
    bool want_borrow_drv_from_ker = true && sz_ker_cur > tr::ker_prb_size_min
            && sz_drv_cur < sz_drv_min;
    if (want_borrow_drv_from_ker) {
        size_t sz_want_borrow = utils::div_up(sz_drv_min, sz_drv_cur);
        for (; prb.nodes[kdims - 1].n % sz_want_borrow; ++sz_want_borrow)
            ;
        if (sz_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / sz_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            printf("split: ");
            prb_dump(prb);
            printf("ndims_ker_max = %d\n", ndims_ker_max);
        });
    }
}

struct jit_uni_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
            const memory_desc_t *imd = input_pd->desc();
            const memory_desc_t *omd = output_pd->desc();

            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *imd, *omd, attr);
            if (prb_init_status != success)
                return prb_init_status;

            DEBUG({
                printf("init : ");
                prb_dump(prb);
            });
            prb_normalize(prb);
            DEBUG({
                printf("norm : ");
                prb_dump(prb);
            });
            prb_simplify(prb);
            DEBUG({
                printf("smpl : ");
                prb_dump(prb);
            });

            prb_block_for_cache(prb);

            int ndims_ker_max;
            prb_thread_kernel_balance(prb, ndims_ker_max);

            tr::kernel_t::desc_t ker_desc;
            status_t ker_init_status
                    = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
            if (ker_init_status != status::success)
                return ker_init_status;

            const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
            if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
                return status::unimplemented;

            DEBUG({
                printf("ker  : ");
                prb_dump(ker_desc.prb);
            });

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr)
                return out_of_memory;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->prb_ = prb;
            _pd->ker_desc_ = ker_desc;
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
    };

    jit_uni_reorder_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        kernel_ = tr::kernel_t::create(pd()->ker_desc_);
        assert(kernel_);
    }
    ~jit_uni_reorder_t() { delete kernel_; }

    void omp_driver_0d(
            int off, const char *in, char *out, const float *scale) const {
        tr::call_param_t c{ in, out, scale };
        (*kernel_)(&c);
#if DEBUG_JIT_UNI_REORDER
	    DBG_MSG_JIT_REORDER(omp driver 0d c.in, c.in);
	    DBG_MSG_JIT_REORDER(omp driver 0d c.out, c.out);
	    DBG_MSG_JIT_REORDER(omp driver 0d c.scale, c.scale);
	    DBG_MSG_JIT_REORDER(omp driver 0d c.in addr, &(c.in));
	    DBG_MSG_JIT_REORDER(omp driver 0d c.out addr, &(c.out));
	    DBG_MSG_JIT_REORDER(omp driver 0d c.scale addr, &(c.scale));
#endif
    }

    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is * data_type_size(pd()->prb_.itype);
            c.out = out + d0 * ns[0].os * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss;
#if DEBUG_JIT_UNI_REORDER
	    DBG_MSG_JIT_REORDER(omp driver 1d c.in, c.in);
	    DBG_MSG_JIT_REORDER(omp driver 1d c.out, c.out);
	    DBG_MSG_JIT_REORDER(omp driver 1d c.scale, c.scale);
	    DBG_MSG_JIT_REORDER(omp driver 1d c.in addr, &(c.in));
	    DBG_MSG_JIT_REORDER(omp driver 1d c.out addr, &(c.out));
	    DBG_MSG_JIT_REORDER(omp driver 1d c.scale addr, &(c.scale));
#endif
            (*kernel_)(&c);
        });
    }

    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d1, ptrdiff_t d0) {
                    auto c = tr::call_param_t();
                    c.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is)
                                    * data_type_size(pd()->prb_.itype);
                    c.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os)
                                    * data_type_size(pd()->prb_.otype);
                    c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss;
#if DEBUG_JIT_UNI_REORDER
	    DBG_MSG_JIT_REORDER(omp driver 2d c.in, c.in);
	    DBG_MSG_JIT_REORDER(omp driver 2d c.out, c.out);
	    DBG_MSG_JIT_REORDER(omp driver 2d c.scale, c.scale);
	    DBG_MSG_JIT_REORDER(omp driver 2d c.in addr, &(c.in));
	    DBG_MSG_JIT_REORDER(omp driver 2d c.out addr, &(c.out));
	    DBG_MSG_JIT_REORDER(omp driver 2d c.scale addr, &(c.scale));
#endif
	(*kernel_)(&c);
	       });
    }

    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
                (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                    auto c = tr::call_param_t();
                    c.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                                    * data_type_size(pd()->prb_.itype);
                    c.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                                    * data_type_size(pd()->prb_.otype);
                    c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss
                            + d2 * ns[2].ss;
#if DEBUG_JIT_UNI_REORDER
	    DBG_MSG_JIT_REORDER(omp driver 3d c.in, c.in);
	    DBG_MSG_JIT_REORDER(omp driver 3d c.out, c.out);
	    DBG_MSG_JIT_REORDER(omp driver 3d c.scale, c.scale);
	    DBG_MSG_JIT_REORDER(omp driver 3d c.in addr, &(c.in));
	    DBG_MSG_JIT_REORDER(omp driver 3d c.out addr, &(c.out));
	    DBG_MSG_JIT_REORDER(omp driver 3d c.scale addr, &(c.scale));
#endif
                    (*kernel_)(&c);
                });
    }

    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
                (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                    auto c = tr::call_param_t();
                    c.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                                      + d3 * ns[3].is)
                                    * data_type_size(pd()->prb_.itype);
                    c.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                                      + d3 * ns[3].os)
                                    * data_type_size(pd()->prb_.otype);
                    c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss
                            + d2 * ns[2].ss + d3 * ns[3].ss;
#if DEBUG_JIT_UNI_REORDER
	    DBG_MSG_JIT_REORDER(omp driver 4d c.in, c.in);
	    DBG_MSG_JIT_REORDER(omp driver 4d c.out, c.out);
	    DBG_MSG_JIT_REORDER(omp driver 4d c.scale, c.scale);
	    DBG_MSG_JIT_REORDER(omp driver 4d c.in addr, &(c.in));
	    DBG_MSG_JIT_REORDER(omp driver 4d c.out addr, &(c.out));
	    DBG_MSG_JIT_REORDER(omp driver 4d c.scale addr, &(c.scale));
#endif
                    (*kernel_)(&c);
                });
    }

    void omp_driver(const char *in, char *out, const float *scale) const {
        in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
        out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

        DEBUG({
            printf("prb : ");
            tr::prb_dump(pd()->prb_);
        });
        DEBUG({
            printf("ker : ");
            tr::prb_dump(pd()->ker_desc_.prb);
        });

        int ndims = pd()->prb_.ndims;
        int ndims_ker = pd()->ker_desc_.prb.ndims;
        assert(ndims - ndims_ker <= ndims_driver_max);

        if (ndims - ndims_ker == 0) {
            set_rnd_mode(pd()->attr()->round_mode_);
            omp_driver_0d(ndims_ker, in, out, scale);
            restore_rnd_mode();
        } else {
            parallel(0, [&](const int ithr, const int nthr) {
                set_rnd_mode(pd()->attr()->round_mode_);
                switch (ndims - ndims_ker) {
                case 1:
                    omp_driver_1d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                case 2:
                    omp_driver_2d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                case 3:
                    omp_driver_3d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                case 4:
                    omp_driver_4d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                default: assert(!"unimplemented");
                }
                restore_rnd_mode();
            });
        }
    }

    virtual void execute(event_t *e) const {
        auto in = reinterpret_cast<const char *>(input_memory(0));
        auto out = reinterpret_cast<char *>(memory());

        omp_driver(in, out, pd()->attr()->output_scales_.scales_);

        e->set_state(event_t::ready);
    }

    enum { ndims_driver_max = 4 };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    tr::kernel_t *kernel_;
};

status_t jit_uni_reorder_create(reorder_pd_t **reorder_pd,
        const memory_pd_t *input_pd, const memory_pd_t *output_pd,
        const primitive_attr_t *attr) {
    return jit_uni_reorder_t::pd_t::create(
            reorder_pd, input_pd, output_pd, attr);
}

} // namespace cpu
} // namespace impl
} // namespace mkldnn
