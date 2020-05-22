/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef JIT_SVE_1x1_CONV_UTILS_HPP
#define JIT_SVE_1x1_CONV_UTILS_HPP

#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#define push(X); \
  CGA64::sub(sp, sp, 8); \
  CGA64::str(X, xa::ptr(sp));

#define pop(X); \
  CGA64::ldr(X, xa::ptr(sp)); \
  CGA64::add(sp, sp, 8);

#define ADDMAX  4095
#define MOVMAX 65535

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

struct reduce_to_unit_stride_t {
    convolution_desc_t conv_d_;
    bool reduce_src_;
    size_t space_per_thread_;
};

/* 1x1-kernel does not support non-unit strides so far, so the idea is:
 *  - for fwd or bwd_weights: to copy src to a scratch memory (with strides
 *    equal to 1) and then call the kernel
 *  - for bwd_data: reduce the problem to the one with unit stride by
 *    performing computations in a scratch memory (with strides equal to 1)
 *    and then copy the result to diff_src */
template <typename conv_pd_t>
inline void rtus_prepare(conv_pd_t *self, const convolution_desc_t *&conv_d,
        const memory_desc_t *&src_d, const memory_desc_t *dst_d) {
    const bool is_bwd_data = self->desc()->prop_kind
        == prop_kind::backward_data;

    const int ndims = src_d->ndims;
    bool rtus_applicable = true
        && utils::pick(ndims - 3,
            (conv_d->strides[0] != 1 && !one_of(conv_d->src_desc.data_type,
                data_type::s16, data_type::bf16, data_type::s32)),
            (conv_d->strides[0] != 1 || conv_d->strides[1] != 1))
        && utils::one_of(src_d->format, memory_format::nCw8c,
            memory_format::nCw16c, memory_format::nChw8c,
            memory_format::nChw16c);
    for (int d = 2; d < ndims; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable
            && conv_d->padding[0][d - 2] == 0
            && dst_d->dims[d] * conv_d->strides[d - 2] == src_d->dims[d];
    }

    if (rtus_applicable) {
        self->rtus_.reduce_src_ = true;
        conv_d = &(self->rtus_.conv_d_ = *conv_d);
        self->rtus_.conv_d_.strides[0] = 1;
        if (ndims == 4)
            self->rtus_.conv_d_.strides[1] = 1;
        utils::array_set(self->rtus_.conv_d_.padding[0], 0, 2);
        if (ndims == 4)
            utils::array_set(self->rtus_.conv_d_.padding[1], 0, 2);
        const int ic = src_d->dims[1];
        if (is_bwd_data) {
            src_d = &(self->rtus_.conv_d_.diff_src_desc = *src_d);
            self->rtus_.conv_d_.diff_src_desc.dims[1] = ic;
            self->rtus_.conv_d_.diff_src_desc.dims[2] = dst_d->dims[2];
            if (ndims == 4)
                self->rtus_.conv_d_.diff_src_desc.dims[3] = dst_d->dims[3];
            memory_desc_wrapper::compute_blocking(
                    self->rtus_.conv_d_.diff_src_desc);
        } else {
            data_type_t data_type = self->rtus_.conv_d_.src_desc.data_type;
            src_d = &(self->rtus_.conv_d_.src_desc = *dst_d);
            self->rtus_.conv_d_.src_desc.dims[1] = ic;
            self->rtus_.conv_d_.src_desc.data_type = data_type;
            memory_desc_wrapper::compute_blocking(
                    self->rtus_.conv_d_.src_desc);
        }
    }
}

template <typename conv_pd_t>
inline void rtus_prepare_space_info(conv_pd_t *self,
        memory_tracking::registrar_t &scratchpad) {
    const auto &jcp = self->jcp_;

    const int max_threads = mkldnn_get_max_threads();
    const size_t factor = utils::pick_by_prop_kind(self->desc()->prop_kind,
            jcp.nb_reduce, jcp.nb_load_blocking_max, jcp.nb_bcast_blocking);
    size_t typesize = types::data_type_size(
            conv_prop_agnostic_src_d(self->desc())->data_type);

    self->rtus_.space_per_thread_ = factor * jcp.is * jcp.ic_block;

    scratchpad.book(memory_tracking::names::key_conv_rtus_space,
            typesize * max_threads * self->rtus_.space_per_thread_);
}

template <cpu_isa_t isa>
struct rtus_driver_t: public jit_generator {

    struct call_params_t {
        const void *ws; /* reduced image (w/ strides = 1) */
        const void *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    void (*ker_)(const call_params_t *p);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(rtus_driver_t)

    using reg64_t = const xa::XReg;
    using zreg_t  = const xa::ZReg;

    const xa::PReg reg_p_all_ones  = p1;

    reg64_t   reg_ws          = x16; // abi_param1;
    reg64_t   reg_src         = x17; // abi_not_param1;
    reg64_t   reg_icb         = x18; // rdx;
    reg64_t   reg_os          = x19; // r11;
    reg64_t   reg_iw_start    = x20; // r8;

    reg64_t   reg_cur_os      = x21; // rax;
    reg64_t   reg_cur_iw      = x22; // r9;
    reg64_t   reg_cur_src     = x23; // r10;

    reg64_t   reg_tmp         = x24; // r10;


    zreg_t reg_zero           = xa::ZReg(0);
    zreg_t reg_v              = xa::ZReg(1);

    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_, vlen_, vlen_shift_;
    bool src_to_ws_;
    size_t typesize_;

    void add_imm(reg64_t out, reg64_t in, int value){
      if( value >= 0){   
        if(value < ADDMAX){
            CGA64::add(out, in, value);
        }else if(value < MOVMAX){
            CGA64::mov(reg_tmp, value);
            CGA64::add(out, in, reg_tmp);
        }else{
            CGA64::mov(reg_tmp, value&0xffff);
            CGA64::movk(reg_tmp, value>>16, 16);
            CGA64::add(out, in, reg_tmp);
        }
      }else{
        int val = -1 * value;
        if(val < ADDMAX){
            CGA64::sub(out, in, val);
        }else if(val < MOVMAX){
            CGA64::mov(reg_tmp, val);
            CGA64::sub(out, in, reg_tmp);
        }else{
            CGA64::mov(reg_tmp, val&0xffff);
            CGA64::movk(reg_tmp, val>>16, 16);
            CGA64::sub(out, in, reg_tmp);
        }

      }
    }


    rtus_driver_t(int iw, int stride_w, int src_step_h,
            int src_step_icb, int ws_step_icb, bool src_to_ws, size_t typesize)
        : iw_(iw), stride_w_(stride_w), src_step_h_(src_step_h)
        , src_step_icb_(src_step_icb), ws_step_icb_(ws_step_icb)
        , src_to_ws_(src_to_ws), typesize_(typesize)
    {
#if 0
        auto Vreg = [=](int idx, int typesize) {
            ZReg res;
            switch (isa) {
            case sve:
                switch (typesize) {
                case 4: res = ZReg(idx); break;
                case 2: assert(!"Not supported 16-bit ver rtus"); break;
                case 1: assert(!"Not supported 8-bit ver rtus"); break;
                default:
                    assert(!"Not supported typesize");
                }
            default:
              assert(!"Not supported vector type");
            }
            return res;
        };

        reg_zero = Vreg(0, typesize);
        reg_v = Vreg(1, typesize);
#endif
        vlen_ = reg_v.getBit() / 8;
        vlen_shift_ = 0;

        int tvlen = vlen_;
        while (tvlen > 1) {
            tvlen /= 2;
            vlen_shift_++;
        }
        generate();
    }

    void loop_is() {

        CGA64::mov(reg_cur_src, reg_src);
        CGA64::mov(reg_cur_iw, reg_iw_start);
        CGA64::mov(reg_cur_os, reg_os);

        xa::LabelAArch64 is_loop;
        CGA64::L_aarch64(is_loop);

        if (src_to_ws_) {
            CGA64::ldr(reg_v, xa::ptr(reg_cur_src));
            CGA64::str(reg_v, xa::ptr(reg_ws));
        } else {
            CGA64::ldr(reg_v, xa::ptr(reg_ws));
            CGA64::str(reg_v, xa::ptr(reg_cur_src));
            for (int w = 1; w < stride_w_; ++w){
                int ofs = w * vlen_;
                ofs = ofs>>6;
                assert( ofs < 256 );
                CGA64::str(reg_zero, xa::ptr(reg_cur_src, ofs));
            }
        }

        add_imm(reg_ws, reg_ws, vlen_);
        add_imm(reg_cur_src, reg_cur_src, stride_w_ * vlen_);

        // for 1d or stride_h=1 convolutions the loop over h should be skipped
        if (!(src_step_icb_ == iw_ || src_step_h_ == iw_)) {
            xa::LabelAArch64 skip_h_step;
            add_imm(reg_cur_iw, reg_cur_iw, stride_w_);
            CGA64::cmp(reg_cur_iw, iw_);
            CGA64::b(xa::LT, skip_h_step);

            if (src_to_ws_) {
                add_imm(reg_cur_src, reg_cur_src, (src_step_h_ - iw_) * vlen_);
            } else {
                reg64_t reg_cur_src_fin = reg_cur_iw; /* just reuse */
                CGA64::mov(reg_cur_src_fin, reg_cur_src);
                add_imm(reg_cur_src_fin, reg_cur_src_fin, (src_step_h_ - iw_) * vlen_);
                xa::LabelAArch64 ih_loop;
                CGA64::L_aarch64(ih_loop);

                for (int w = 0; w < stride_w_; ++w){
                    int ofs = w * vlen_;
                    ofs = ofs>>6;
                    assert( ofs < 256 );

                    CGA64::str(reg_zero, xa::ptr(reg_cur_src, ofs));
                }

                add_imm(reg_cur_src, reg_cur_src, stride_w_ * vlen_);
                CGA64::cmp(reg_cur_src, reg_cur_src_fin);
                CGA64::b(xa::LT, ih_loop);
            }
            CGA64::mov(reg_cur_iw, 0);
            CGA64::L_aarch64(skip_h_step);
        }
        assert(vlen_ < 4096);
        CGA64::subs(reg_cur_os, reg_cur_os, vlen_);
        CGA64::b(xa::NE, is_loop);

        /* restore dst */
        CGA64::sub(reg_ws, reg_ws, reg_os);
    }

    void generate() {
        assert( isa == sve );

        preamble();

#if defined(_WIN32)
        assert(reg_src == xa::abi_not_param1 && xa::abi_not_param1 == rdi);
        push(rdi);
#endif

#define READ_PARAM(what) \
        CGA64::ldr(reg_ ## what, xa::ptr(abi_param1_aarch64, static_cast<int32_t>(offsetof(call_params_t, what))))

        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);

        READ_PARAM(ws); /* reg_ws should always be read the last */
#undef  READ_PARAM

        CGA64::lsl(reg_os, reg_os, vlen_shift_);

        if (!src_to_ws_) {
            switch (reg_zero.getBit() / 8) {
            case 16 /*xmm*/:
                assert(!"rtus kernel failure (128-bit)");
                break;
            case 32 /*ymm*/:
                {
                assert(!"rtus kernel failure (256-bit)");
                break;
                }
            case 64 /*zmm*/:
                {
                xa::ZRegS zreg_zs(reg_zero.getIdx());
                CGA64::fmov(zreg_zs);
                break;
                }
            default: assert(!"rtus kernel failure");
            }
        }

        xa::LabelAArch64 icb_loop;
        CGA64::L_aarch64(icb_loop);


        loop_is();

        add_imm(reg_ws, reg_ws, ws_step_icb_ * vlen_);
        add_imm(reg_src, reg_src, src_step_icb_ * vlen_);

        CGA64::subs(reg_icb, reg_icb, 1); //dec(reg_icb);
        CGA64::b(xa::NE, icb_loop);

#if defined(_WIN32)
        pop(rdi);
#endif

        uni_vzeroupper(); // jit_generator_aarch64 

        postamble();

        this->ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint32_t*>(this->getCode32()));
    }
};

template <cpu_isa_t isa, typename conv_t>
inline void init_rtus_driver(conv_t *self) {
    const auto &conf = *self->pd();
    if (!conf.rtus_.reduce_src_) return;

    const auto &cd = *conf.desc();
    const int ndims = conf.ndims();
    const int stride_h = (conf.ndims() == 3) ? 1 : cd.strides[0];
    const int stride_w = cd.strides[ndims - 3];

    const bool is_bwd_data = cd.prop_kind == prop_kind::backward_data;
    const auto &src_d = is_bwd_data ? *conf.diff_src_pd()->desc()
                                    : *conf.src_pd()->desc();
    assert((isa == sve && utils::one_of(
            src_d.format, memory_format::nCw16c, memory_format::nChw16c)));

    const int ih = ndims == 3 ? 1 : src_d.dims[2];
    const int iw = src_d.dims[ndims - 1];

    const int src_step_h = stride_h * iw;
    const int src_step_icb = ih * iw;
    const int ws_step_icb = conf.jcp_.is;
    const bool src_to_ws = !is_bwd_data;
    const size_t typesize = types::data_type_size(
            conv_prop_agnostic_src_d(self->pd()->desc())->data_type);

    self->rtus_driver_ = new rtus_driver_t<isa>(iw, stride_w, src_step_h,
            src_step_icb, ws_step_icb, src_to_ws, typesize);
}

inline int best_divider(int value, int min_divider, int max_divider,
        bool find_max, int step = 1)
{
    max_divider = nstl::max(1, nstl::min(max_divider, value));
    min_divider = nstl::max(1, nstl::min(min_divider, max_divider));

    auto loss_ratio = [](int total, int chunk)
    { return float(rnd_up(total, chunk) - total) / rnd_up(total, chunk); };

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

}
}
}

#endif
