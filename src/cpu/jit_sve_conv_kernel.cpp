/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_barrier.hpp"
#include "cpu_memory.hpp"

#include "jit_sve_conv_kernel.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_conv_call_s, field))
#define KNx_L2_EFFECTIVE_CAPACITY ((512-64)*1024)

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

#define push(X); \
    CGA64::sub(CGA64::sp, CGA64::sp, 8); \
    CGA64::str(X, xa::ptr(CGA64::sp));

#define pop(X); \
    CGA64::ldr(X, xa::ptr(CGA64::sp)); \
    CGA64::add(CGA64::sp, CGA64::sp, 8);



namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;


namespace {

constexpr auto small_spatial = 14;
unsigned int L1_cache_size = get_A64FX_cache_size(1, true, 1);


inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // ow-threading is currently implemented for forward only
    // TODO: single code for fwd and bwd after ow-thr for bwd
    // meaningless switch was removed
    if (jcp.prop_kind == backward_data) {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cgn : loop_gnc;
    } else {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cwgn : loop_gncw;
    }
}

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    if (mayiuse(sve))
        return (jcp.ic < 16 && jcp.ngroups == 1);
    else
        return one_of(jcp.ic, 1, 3);
}

inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}

inline bool is_owb_prefetching(const jit_conv_conf_t &jcp) {
    return (jcp.ver == ver_4fma && is_ow_threading_on(jcp));
}

}

template<typename Vmm>
void _jit_sve_conv_fwd_kernel<Vmm>::prepare_output(int ur_w)
{
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };

    for (int k = 0; k < jcp.nb_oc_blocking; k++){
        for (int j = 0; j < ur_w; j++) {

            CGA64::fmov(zreg_out_s(j, k));
#if 0
            if (!is_owb_prefetching(jcp)) {
                size_t aux_output_offset = get_output_offset(j, k);
                mic_prefetcht1(EVEX_compress_addr_safe(reg_out_prf,
                            aux_output_offset, reg_out_long_offt));
            }
#endif 
        }
    }
}

template<typename Vmm>
void _jit_sve_conv_fwd_kernel<Vmm>::store_output(int ur_w)
{

    auto zreg_tmp = [=](int idx){
        return xa::ZReg(idx);
    };
    auto zreg_tmp_s = [=](int idx){
        return xa::ZRegS(idx);
    };

    auto zreg_out = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZReg(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };

    xa::LabelAArch64 no_update_label, store_label, eltwise_label;

    CGA64::ldr(reg_channel, xa::ptr(abi_param1_aarch64, GET_OFF(channel)));
    if (jcp.with_bias) {
        CGA64::ldr(reg_bias, xa::ptr(abi_param1_aarch64, GET_OFF(bias)));
    }

    if (!jcp.with_sum) {
        CGA64::cmp(reg_channel, 0);
        CGA64::b(xa::EQ, no_update_label);
    }

    int reg_ofs = jcp.ur_w * jcp.nb_oc_blocking;
    int num_regs = 32 - reg_ofs;

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = get_output_offset(j, k);
            int idx = reg_ofs + ((j + k * ur_w)%num_regs);
            add_imm(reg_out_long_offt, reg_out, aux_output_offset);
            CGA64::ldr(zreg_tmp(idx), xa::ptr(reg_out_long_offt));
            CGA64::fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(idx));
        }

    if (!jcp.with_sum) {
        CGA64::b(eltwise_label);
    } else {
        CGA64::cmp(reg_channel, 0);
        CGA64::b(xa::NE, eltwise_label);
    }

    auto bias_load = [=] (int bias_offset, int idx){
        int ofs = bias_offset;
        
        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::ldr(zreg_tmp(idx), xa::ptr(reg_bias, static_cast<int32_t>(ofs)));
        }else{
            add_imm(reg_tmp_addr, reg_bias, ofs); 
            CGA64::ldr(zreg_tmp(idx), xa::ptr(reg_tmp_addr));
        }
    };

    CGA64::L_aarch64(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                int idx = reg_ofs + ((j + k * ur_w)%num_regs);
                bias_load(bias_offset, idx);
                CGA64::fadd(zreg_out_s(j,k), zreg_out_s(j,k), zreg_tmp_s(idx));
            }
        }
    }

    CGA64::L_aarch64(eltwise_label);
    if (jcp.with_eltwise) {
        assert(!jcp.with_eltwise);
#if 0
        cmp(reg_channel, jcp.nb_ic - 1);
        b(LT, store_label);

        {
            if (ur_w == jcp.ur_w) {
                eltwise_injector_->compute_vector_range(0,
                        jcp.nb_oc_blocking * jcp.ur_w);
            } else {
                for (int k = 0; k < jcp.nb_oc_blocking; k++)
                    eltwise_injector_->compute_vector_range(k * jcp.ur_w,
                            k * jcp.ur_w + ur_w);
            }
        }
#endif
    }

    auto out_str = [=](int j, int k, int aux_output_offset){
        int ofs = aux_output_offset;
        
        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::str(zreg_out(j, k), xa::ptr(reg_out, static_cast<int32_t>(ofs)));
        }else{ 
            add_imm(reg_tmp_addr, reg_out, ofs);
            CGA64::str(zreg_out(j, k), xa::ptr(reg_tmp_addr));
        }
    };

    CGA64::L_aarch64(store_label);
    for (int k = 0; k < jcp.nb_oc_blocking; k++){
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = (size_t)typesize *
                ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;

            out_str(j, k, aux_output_offset);
        }
    }
}

template<typename Vmm>
void _jit_sve_conv_fwd_kernel<Vmm>::compute_loop_fma_core(int ur_w,
    int pad_l, int pad_r)
{
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    xa::LabelAArch64 kh_label, kd_label;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * jcp.oc_block
        * jcp.ic_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
        * inp_mul;


    auto input_offset = [=](int oi, int ic, int ki) {
        return (size_t)jcp.typesize_in
                * ((size_t)(ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                * inp_mul + (size_t)ic
                * (!jcp.is_1stconv ? 1 : (size_t)jcp.iw * jcp.ih * jcp.id));
    };


    if (one_of(jcp.ndims, 3, 4)) {
        CGA64::mov(aux_reg_inp, reg_inp);
        CGA64::mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_out);

        CGA64::ldr(reg_ki, xa::ptr(abi_param1_aarch64, GET_OFF(kd_padding)));
        CGA64::ldr(aux_reg_ker_d, xa::ptr(abi_param1_aarch64, GET_OFF(filt)));
        CGA64::mov(aux_reg_inp_d, reg_inp);

        CGA64::L_aarch64(kd_label);
        CGA64::ldr(reg_kj, xa::ptr(abi_param1_aarch64, GET_OFF(kh_padding)));
    } else {
        CGA64::mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        CGA64::mov(aux_reg_inp, aux_reg_inp_d);
        CGA64::mov(aux_reg_ker, aux_reg_ker_d);
    }

    auto zreg_inp = [=](int i_ic, int nb_x_blocking){
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return xa::ZReg(idx);
    };
    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking){
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return xa::ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };
    auto zreg_wei = [=](int idx){
        assert(idx < 32);
        return xa::ZReg(idx);
    };
    auto zreg_wei_s = [=](int idx){
        assert(idx < 32);
        return xa::ZRegS(idx);
    };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_input_offset, int prev_ofs, int jj_end){
        if( ((aux_input_offset & 0x3) ==0) && 
                (aux_input_offset < LDRWMAX) &&
                (aux_input_offset >= 0)){
            CGA64::ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                    xa::ptr(aux_reg_inp, static_cast<int32_t>(aux_input_offset)));
        }else{
            if( (prev_ofs != -1) && 
                ((aux_input_offset - prev_ofs)>0) &&
                ((aux_input_offset - prev_ofs) < LDRWMAX) && 
                (((aux_input_offset - prev_ofs)& 0x3) ==0)){
                
                CGA64::ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                        xa::ptr(reg_prev_bcast_addr, static_cast<int32_t>(aux_input_offset - prev_ofs)));
            }else{
                int ofs;
                if((prev_ofs != -1) && ((aux_input_offset - prev_ofs)>0)){
                    ofs = aux_input_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs);
                }else{
                    ofs = aux_input_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_inp, ofs);
                }

                CGA64::ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr));
                prev_ofs = aux_input_offset;
            }
        }

        return prev_ofs;

    };

    auto wei_load = [=](int aux_kernel_offset, int reg_idx, int prev_ofs){
        int ofs = aux_kernel_offset;

        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::ldr(zreg_wei(reg_idx),
                         xa::ptr(aux_reg_ker, static_cast<int32_t>(ofs)));
        }else{
            int ofs_tmp = ofs - prev_ofs;
            ofs_tmp = ofs_tmp >> 6;
            if( (prev_ofs != -1) && (ofs_tmp>0) &&
                (ofs_tmp < LDRMAX) ){
                CGA64::ldr(zreg_wei(reg_idx), 
                       xa::ptr(reg_prev_wei_addr, static_cast<int32_t>(ofs_tmp)));
            }else{
                if((prev_ofs != -1) && (ofs_tmp>0)){
                    ofs_tmp = aux_kernel_offset - prev_ofs;
                    add_imm(reg_prev_wei_addr, reg_prev_wei_addr, ofs_tmp);
                }else{
                    add_imm(reg_prev_wei_addr, aux_reg_ker, ofs);
                }

                CGA64::ldr(zreg_wei(reg_idx), xa::ptr(reg_prev_wei_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;

    };

    CGA64::L_aarch64(kh_label);
    {
        int prev_bcast_ofs = -1;
        int prev_wei_ofs = -1;
        for (int ki = 0; ki < kw; ki++) {

            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);

            int wei_reg_ofs = nb_oc_block * jcp.ur_w + jj_end;
            int num_regs4wei = 32 - wei_reg_ofs;
            for (int ic = 0; ic < ic_block; ic++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        size_t aux_input_offset = input_offset(jj, ic, ki);
                        prev_bcast_ofs = bcast_load(jj, nb_oc_block, aux_input_offset, prev_bcast_ofs, jj_end);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int reg_idx = wei_reg_ofs + ii;
                    if(reg_idx >= 32) break;
                    int aux_kernel_offset = jcp.typesize_in
                        * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd * ic_block
                        * oc_block + ki * ic_block * oc_block + ic * oc_block);

                    wei_count ++;
                    if (jj_end - jj_start > 0){
                        prev_wei_ofs = 
                            wei_load(aux_kernel_offset, 
                                     wei_reg_ofs +(ii%num_regs4wei), 
                                     prev_wei_ofs);
                    }
                }

                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                        * ((ii+wei_count) * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd * ic_block
                        * oc_block + ki * ic_block * oc_block + ic * oc_block);

                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == expl_bcast)
                            CGA64::fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                        zreg_inp_s(jj, nb_oc_block), 
                                        zreg_wei_s(wei_reg_ofs +(ii%num_regs4wei)));
                        else 
                            assert(NULL);

                    if ((jj_end - jj_start > 0) && ((wei_count + ii) < nb_oc_block)){
                        prev_wei_ofs =
                            wei_load(aux_kernel_offset, 
                                     wei_reg_ofs +((ii+wei_count)%num_regs4wei),
                                     prev_wei_ofs);
                    }
                }
            }
        }
        add_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr);
        add_imm(aux_reg_inp, aux_reg_inp, shift_input_ptr);
        CGA64::sub(reg_kj, reg_kj, 1); //dec(reg_kj);
        CGA64::cmp(reg_kj, 0);
        CGA64::b(xa::GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d, aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add_imm(aux_reg_ker_d, aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        CGA64::sub(reg_kj, reg_kj, 1); //dec(reg_ki);
        CGA64::cmp(reg_ki, 0);
        CGA64::b(xa::GT, kd_label);

        pop(reg_out);
    }

}

template<typename Vmm>
void _jit_sve_conv_fwd_kernel<Vmm>::compute_loop(int ur_w,
        int pad_l, int pad_r)
{

    if (jcp.ndims == 5){
        push(reg_oi);
    }
    prepare_output(ur_w);

    xa::LabelAArch64 skip_compute_loop;
    if (jcp.ndims == 5) {
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            CGA64::ldr(reg_kj, xa::ptr(abi_param1_aarch64, GET_OFF(kd_padding)));
            CGA64::cmp(reg_kj, 0);
            CGA64::b(xa::LE, skip_compute_loop);
        }
    }
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        CGA64::ldr(reg_kj, xa::ptr(abi_param1_aarch64, GET_OFF(kh_padding)));
        CGA64::cmp(reg_kj, 0);
        CGA64::b(xa::LE, skip_compute_loop);
    }

    if (jcp.ver == ver_fma)
        if (jcp.is_1stconv && jcp.kernel_kind != expl_bcast)
            assert(NULL);
        else
            if (jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking == 1)
                assert(jcp.kernel_kind != embd_bcast);
            else
                compute_loop_fma_core(ur_w, pad_l, pad_r);
    else
        assert(!"unknown convolution version");

    CGA64::L_aarch64(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) {
        pop(reg_oi);
    }

}

template<typename Vmm>
void _jit_sve_conv_fwd_kernel<Vmm>::generate()
{
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * ur_w * stride_w * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;
    int out_shift = jcp.typesize_out * ur_w * jcp.oc_block;

    preamble();
    CGA64::ldr(reg_inp,     xa::ptr(abi_param1_aarch64, GET_OFF(src)));
    CGA64::ldr(reg_out,     xa::ptr(abi_param1_aarch64, GET_OFF(dst)));
    CGA64::ldr(reg_ker,     xa::ptr(abi_param1_aarch64, GET_OFF(filt)));
    CGA64::ldr(reg_ker_prf, xa::ptr(abi_param1_aarch64, GET_OFF(filt_prf)));
    CGA64::ldr(reg_kh,      xa::ptr(abi_param1_aarch64, GET_OFF(kh_padding)));

    int r_pad = nstl::max(
            0, (ow - 1) * stride_w + (kw - 1) * dilate_w - (iw + l_pad - 1));
    int n_oi = ow / ur_w;
    int r_pad1 = (ur_w * n_oi - 1) * stride_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);

    CGA64::ptrue( reg_p_all_ones.b );

    if (!is_ow_threading_on(jcp)) { // nb_ow <= 1
        // nb_ow is # of output width blocks ??

        // ow is being processed as a whole - with left and right paddings
        // n_oi is # of output width blocks ??
        if (r_pad1 > 0) n_oi--;

        if (ow == ur_w) {
            CGA64::ldr(reg_inp_prf, xa::ptr(abi_param1_aarch64, GET_OFF(src_prf)));
            CGA64::ldr(reg_out_prf, xa::ptr(abi_param1_aarch64, GET_OFF(dst_prf)));
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            CGA64::mov(reg_inp_prf, reg_inp);
            CGA64::mov(reg_out_prf, reg_out);
            if (n_oi == 0) {
                add_imm(reg_inp_prf, reg_inp_prf,inp_shift_pad);
                add_imm(reg_out_prf, reg_out_prf, out_shift);
                compute_loop(ur_w, l_pad, r_pad1);
                add_imm(reg_inp, reg_inp, inp_shift_pad);
                add_imm(reg_out, reg_out, out_shift);
                if (ur_w_tail != 0) {
                    add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
                    add_imm(reg_out_prf, reg_out_prf, out_shift);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            } else {
                CGA64::mov(reg_oi, 0); 
                if (l_pad > 0) {
                    add_imm(reg_inp_prf, reg_inp_prf, inp_shift_pad);
                    add_imm(reg_out_prf, reg_out_prf, out_shift);
                    compute_loop(ur_w, l_pad, 0);
                    add_imm(reg_inp, reg_inp,inp_shift_pad);
                    add_imm(reg_out, reg_out,out_shift);
                    add_imm(reg_oi, reg_oi, 1); // increment
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    xa::LabelAArch64 ow_loop_label;
                    CGA64::L_aarch64(ow_loop_label);
                    {
                        add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
                        add_imm(reg_out_prf, reg_out_prf, out_shift);
                        compute_loop(ur_w, 0, 0);
                        add_imm(reg_inp, reg_inp, inp_shift);
                        add_imm(reg_out, reg_out, out_shift);
                        add_imm(reg_oi, reg_oi, 1); //inc(reg_oi);
                        CGA64::cmp(reg_oi, n_oi);
                        CGA64::b(xa::LT, ow_loop_label);
                    }
                }
                if (r_pad1 > 0) {
                    add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
                    add_imm(reg_out_prf, reg_out_prf, out_shift);
                    compute_loop(ur_w, 0, r_pad1);
                    add_imm(reg_inp, reg_inp, inp_shift);
                    add_imm(reg_out, reg_out, out_shift);
                }
                if (ur_w_tail != 0) {
                    add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
                    add_imm(reg_out_prf, reg_out_prf, out_shift);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        xa::LabelAArch64 end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        xa::LabelAArch64 oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow-1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded) n_oi_last_ow_block--;
        else if (first_ow_block_padded) n_oi_first_ow_block--;
        else if (next_last_ow_block_padded) n_oi_next_last_ow_block--;

        CGA64::ldr(reg_owb, xa::ptr(abi_param1_aarch64, GET_OFF(owb)));
        CGA64::cmp(reg_owb, 0); // is that the first ow-block ?
        CGA64::b(xa::GT, middle_ow_blocks_label);

        // the first ow block, compute left padding

        CGA64::mov(reg_oi, n_oi_first_ow_block);
        CGA64::mov(reg_inp_prf, reg_inp);
        CGA64::mov(reg_out_prf, reg_out);

        if (l_pad > 0) {
            CGA64::ldr(reg_ker_prf, xa::ptr(abi_param1_aarch64, GET_OFF(filt_prf)));
            add_imm(reg_inp_prf, reg_inp_prf, inp_shift_pad);
            add_imm(reg_out_prf, reg_out_prf, out_shift);
            compute_loop(ur_w, l_pad, 0);
            add_imm(reg_inp, reg_inp, inp_shift_pad);
            add_imm(reg_out, reg_out, out_shift);
            CGA64::sub(reg_oi, reg_oi, 1); // decrement
        }
        CGA64::b(oi_loop_label);

        // middle or last ow block entry

        CGA64::L_aarch64(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add_imm(reg_inp, reg_inp, inp_shift_pad_second_block);
            add_imm(reg_inp_prf, reg_inp_prf, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        CGA64::cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
        CGA64::mov(reg_oi, n_oi_last_ow_block);
        CGA64::b(xa::EQ, oi_loop_label);
        CGA64::cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        CGA64::mov(reg_oi, n_oi_next_last_ow_block);
        CGA64::b(xa::EQ, oi_loop_label);
        CGA64::mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        CGA64::L_aarch64(oi_loop_label);
        CGA64::ldr(reg_ker_prf, xa::ptr(abi_param1_aarch64, GET_OFF(filt_prf)));
        CGA64::L_aarch64(oi_loop_start_label);
            CGA64::cmp(reg_oi, 0);
            CGA64::b(xa::LE, oi_loop_end_label);

            add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
            add_imm(reg_out_prf, reg_out_prf, out_shift);
            compute_loop(ur_w, 0, 0);
            add_imm(reg_inp, reg_inp, inp_shift);
            add_imm(reg_out, reg_out, out_shift);
            CGA64::sub(reg_oi, reg_oi, 1); // dec(reg_oi);
            CGA64::b(oi_loop_start_label);
        CGA64::L_aarch64(oi_loop_end_label);

        CGA64::ldr(reg_owb, xa::ptr(abi_param1_aarch64, GET_OFF(owb)));

        CGA64::cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            CGA64::b(xa::EQ, last_oi_label);
        } else {
            CGA64::b(xa::EQ, end_label);
        }
        CGA64::cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        CGA64::b(xa::LT, end_label);
        if (next_last_ow_block_padded) {
            CGA64::b(xa::EQ, last_oi_label);
        } else {
            CGA64::b(xa::EQ, end_label);
        }
        // that is last block
        if (!last_ow_block_padded) {
            CGA64::b(tail_label);
        }

        // last oi block with right padding
        CGA64::L_aarch64(last_oi_label);
        CGA64::ldr(reg_ker_prf, xa::ptr(abi_param1_aarch64, GET_OFF(filt_prf)));
        add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
        add_imm(reg_out_prf, reg_out_prf, out_shift);
        compute_loop(ur_w, 0, r_pad1);
        add_imm(reg_inp, reg_inp, inp_shift);
        add_imm(reg_out, reg_out, out_shift);

        CGA64::ldr(reg_owb, xa::ptr(abi_param1_aarch64, GET_OFF(owb)));
        CGA64::cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        CGA64::b(xa::LT, end_label);

        CGA64::L_aarch64(tail_label);
        CGA64::ldr(reg_ker_prf, xa::ptr(abi_param1_aarch64, GET_OFF(filt_prf)));
        if (ur_w_tail != 0) {
            add_imm(reg_inp_prf, reg_inp_prf, inp_shift);
            add_imm(reg_out_prf, reg_out_prf, out_shift);
            compute_loop(ur_w_tail, 0, r_pad);
        }
        CGA64::L_aarch64(end_label);
    }
    postamble();

#if 0
    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
#endif
}

bool jit_sve_conv_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
    case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
    default: return false;
    }

    return false;
}

status_t jit_sve_conv_fwd_kernel::init_conf(
            jit_conv_conf_t &jcp, const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd, cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr, int nthreads)
{
    using namespace prop_kind;

    if (!mayiuse(sve))
        return status::unimplemented;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];
    jcp.src_fmt = src_d.format();

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    // ??
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    // Check the lenght of the input channel. Why?
    jcp.is_1stconv = is_1stconv(jcp);

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits<sve>::vlen / sizeof(float);
    jcp.simd_w = full_simd_w; // 512-bit simd

    // Check whethear simd_w should be changed to 128-bit or not.
    bool ok_to_try_128bit = true
        && mayiuse(sve)
        && src_d.data_type() == data_type::f32
        && !jcp.is_1stconv
        && !ok_to_pad_channels
        && (jcp.ic % jcp.simd_w != 0 || jcp.oc % jcp.simd_w != 0)
        && (jcp.ic % 8 != 0 || jcp.oc % 8 != 0);
    if (ok_to_try_128bit){
        jcp.simd_w = 4; // 128-bit simd
        return status::unimplemented;
    }

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;
    jcp.aligned_threads = 0;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    // Check eltwise ops after convolution
    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
#if 1
        return status::unimplemented;
#else
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
#endif
    }

    auto src_format = jcp.is_1stconv
        ? pick(ndims - 3, ncw, nchw, ncdhw)                 // first convolution
        : ((jcp.simd_w == 4)
            ? pick(ndims - 3, nCw4c, nChw4c, nCdhw4c)       // for 128-bit simd
            : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c));  // for 512-bit

    auto dst_format = (jcp.simd_w == 4)
        ? pick(ndims - 3, nCw4c, nChw4c, nCdhw4c)           // for 128-bit
        : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);       // for 512-bit

    auto wei_format = with_groups
        ? ((jcp.simd_w == 4)    // with_groups
            ? pick(ndims - 3, gOIw4i4o, gOIhw4i4o, gOIdhw4i4o)          // for 128-bit
            : pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o))   // for 512-bit
        : ((jcp.simd_w == 4)    // not with_groups
            ? pick(ndims - 3, OIw4i4o, OIhw4i4o, OIdhw4i4o)             // for 128-bit
            : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o));     // for 512-bit

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));
    if (src_d.format() != src_format)
        return status::unimplemented;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(dst_format));
    if (dst_d.format() != dst_format)
        return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    if (mayiuse(sve) &&
            src_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && dst_d.data_type() == data_type::f32) {

        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);

        if (jcp.is_1stconv) {

            const auto w_format = with_groups
                ? ((jcp.simd_w == 4)    // with_groups
                    ? pick(ndims - 3, gOwi4o, gOhwi4o, gOdhwi4o)    // 128-bit
                    : pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o))// 512-bit
                : ((jcp.simd_w == 4)    // not with_groups
                    ? pick(ndims - 3, Owi4o, Ohwi4o, Odhwi4o)       // 128-bit
                    : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o));  // 512-bit

            if (weights_d.format() == any)
                CHECK(weights_pd.set_format(w_format));
            if (weights_d.format() != w_format)
                return status::unimplemented;

        } else {
            if (weights_d.format() == any)
                CHECK(weights_pd.set_format(wei_format));
            if (weights_d.format() != wei_format)
                return status::unimplemented;
        }
    } else {
        return status::unimplemented;
    }

    jcp.ur_w = nstl::min(jcp.ow, regs); // ur_w is min(output width, regs=28)
    // TODO (Tanya): currently applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs/2)
        jcp.ur_w = regs;

    int n_oi = (jcp.ow / jcp.ur_w); // num of blocks of output width
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);
    if (jcp.l_pad > 0 && r_pad > 0)
        n_oi--;

    // Heuristic to optimize code size on KNX
    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) mult += 1;
        if (r_pad > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off
            = (jcp.ngroups > 1 && one_of(src_d.format(), ncw, nchw, ncdhw)) ?
            jcp.ic :
            1;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.ow_block = jcp.ow;

    auto get_thr_eff = [=](int nb_oc_blocking, int ow_block) {
        int nb_ow = div_up(jcp.ow, ow_block);
        int nb_oc_chunks = div_up(jcp.nb_oc, nb_oc_blocking);
        int work_amount = jcp.mb * jcp.oh * nb_oc_chunks * nb_ow;
        float disbalance = (float)jcp.ow / rnd_up(jcp.ow, ow_block);
        float thr_eff = disbalance * (float)work_amount
            / rnd_up(work_amount, nthreads);
        return thr_eff;
    };

    auto is_ow_threading_applicable = [=]() {
        return (true && !jcp.is_1stconv && one_of(jcp.ndims, 3, 4)
                 && IMPLICATION(mayiuse(avx512_mic),
                 jcp.ver == ver_4fma && IMPLICATION(jcp.mb != 1, jcp.ih == 1 && jcp.kh == 1)));
    };

    auto get_ow_block = [=](int nb_oc_blocking, int ur_w, float &eff) {
        int res_ow_block = jcp.ow;
        eff = get_thr_eff(nb_oc_blocking, res_ow_block);
        if (!is_ow_threading_applicable())
            return res_ow_block;

        int L2_part = (get_A64FX_cache_size(2, false, nthreads) * 7 / 8) / typesize;
        if (jcp.ver == ver_4fma)
            L2_part /= 2;
        int size_src_chunk = jcp.ic_block * ur_w * jcp.kh;
        int size_dst_chunk = jcp.oc_block * nb_oc_blocking * ur_w;
        int size_wei_chunk = jcp.oc_block * nb_oc_blocking * jcp.ic_block
            * jcp.kw * jcp.kh;
        int nurw_cache = (L2_part - 2 * size_wei_chunk)
            / (2 * size_dst_chunk + 2 * size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        int ow_block_cache = ur_w * nstl::max(2, nurw_cache);

        int ow_block_thr = ow_block_cache;
        eff = get_thr_eff(nb_oc_blocking, ow_block_thr);

        int max_nb_ow = div_up(jcp.ow, 2 * ur_w);
        int start_nb_ow = div_up(jcp.ow, ow_block_thr);
        for (int nb_ow = start_nb_ow; nb_ow <= max_nb_ow; nb_ow++) {
            int ow_block
                = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), ur_w), jcp.ow);
            float eff_threshold = (jcp.ver == ver_4fma) ? 0.8f : 0.9f; //0.9f
            if (ow_block < nb_oc_blocking * jcp.oc_block && eff > eff_threshold)
                break;
            if (div_up(jcp.ow, ow_block) != nb_ow)
                continue;
            float thr_eff = get_thr_eff(nb_oc_blocking, ow_block);
            float eff_step = (jcp.ver == ver_4fma) ? 1.1f : 1.f;
            if (ow_block >= 2 * ur_w && thr_eff > eff_step * eff) {
                ow_block_thr = ow_block;
                eff = thr_eff;
            }
            eff_threshold = (jcp.ver == ver_4fma) ? 0.9f : 0.98f;
            if (eff > eff_threshold)
                break;
        }
        res_ow_block = nstl::min(jcp.ow, nstl::max(2 * ur_w, ow_block_thr));
        eff = get_thr_eff(nb_oc_blocking, res_ow_block);
        return res_ow_block;
    };


    if (jcp.ver == ver_fma && mayiuse(sve)) {
        int try_nb_oc_blocking = 2;
        unsigned int ker_inp_size = typesize * div_up(jcp.iw, jcp.stride_w)
            * jcp.ic_block * jcp.kh * jcp.kd;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block
            * try_nb_oc_blocking;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_oc_blocking * jcp.kd;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;

#ifdef __ARM_ARCH
        bool embd_bcast_condition = false;
#else
        bool embd_bcast_condition = true
            && (jcp.kw == 3 && jcp.ow <= 28 && ker_total_size < L1_cache_size)
            && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
            && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512);
#endif
        if (jcp.mb == 1) {
            unsigned int inp_size = jcp.mb * div_up(jcp.ih, jcp.stride_h)
                    * div_up(jcp.iw, jcp.stride_w) * jcp.ic;
            unsigned int wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            int job_size_min = work_amount / nthreads;
            int job_size_max = div_up(work_amount, nthreads);
            int ch_max = rnd_up(jcp.oh, job_size_max);
            int ch_min = (job_size_min == 0)
                ? jcp.oh
                : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == loop_cgn && oc_chunks > 1 && nthreads > 1
                && wei_size / inp_size > 24
                && (not_aligned_max || not_aligned_min)
                && eligible_case) {
                // Try to find nthreads > mkldnn_get_max_threads() / 2 such
                // that oc_chunks is a multiple of nthreads, or nthreads is a
                // multiple of oc_chunks. Otherwise, keep default value.
                // TODO: implement a task-based alternative without throttling.
                jcp.aligned_threads = nthreads;
                for (int i = nthreads; i > nthreads / 2; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        }

#ifndef __ARM_ARCH
        if (jcp.kw > 3
                || (jcp.stride_w == 1 && jcp.stride_h == 1
                           && embd_bcast_condition)
                || ((jcp.stride_w != 1 || jcp.stride_h != 1)
                           && ((jcp.mb <= 16 && (jcp.oc <= 192 || jcp.oh <= 10)
                                      && embd_bcast_condition)))
                || (jcp.mb == 1
                           && (jcp.ur_w >= jcp.ow || jcp.is_1stconv
                                      || (jcp.ow <= 147 && jcp.oc <= 96)))) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.ow, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
                    && jcp.kw <= 3 && jcp.nb_oc % try_nb_oc_blocking == 0
                    && IMPLICATION(jcp.is_1stconv, jcp.mb == 1)
                    && IMPLICATION(jcp.mb == 1, jcp.ur_w < jcp.ow)) {
                jcp.nb_oc_blocking = try_nb_oc_blocking;
                jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
            }
        } else 
#endif // ifndef __ARM_ARCH
        {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            if (IMPLICATION(jcp.is_1stconv, jcp.mb >= 1)) {
                float best_thr_eff = 0.f;
                int best_nb_oc_blocking = 1;
                for (int i = nstl::min(jcp.nb_oc, 5); i > 0; i--) {
                    if (jcp.nb_oc % i == 0) {
                        float thr_eff;
                        int ur_w = nstl::min(jcp.ow, 31 / (i + 1));
                        get_ow_block(i, ur_w, thr_eff);
                        if (thr_eff > 1.05f * best_thr_eff) {
                            best_nb_oc_blocking = i;
                            best_thr_eff = thr_eff;
                        }
                    }
                }
                jcp.nb_oc_blocking = best_nb_oc_blocking;
                jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
            }
        }
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    float thr_eff;
    jcp.ow_block = get_ow_block(jcp.nb_oc_blocking, jcp.ur_w, thr_eff);
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    const int L2_size = get_A64FX_cache_size(2, false, nthreads) / sizeof(float);
    // Source and output data needs to fit in L2,
    // leaving some space for weights and prefetching.
    // 0.6f
    int h_L2 = int(((0.6f * L2_size) / jcp.simd_w
                           - nstl::min(0, jcp.kh - jcp.stride_h) * jcp.iw)
            / (jcp.stride_h * jcp.iw + jcp.ow));
    jcp.h_blocking = nstl::max(1, nstl::min(jcp.oh, h_L2));
    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (jcp.l_pad > 0) + (r_pad > 0);
        const float max_instruction_size = 15;
        float ur_fac
                = (float)jcp.kw * jcp.ic_block * jcp.nb_oc_blocking * jcp.ur_w;
        float code_size = mult * ur_fac * max_instruction_size;
        if (code_size > max_code_size) return status::unimplemented;
    }

    return status::success;
}

void jit_sve_conv_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.typesize_out * jcp.oc);
}

void jit_sve_conv_bwd_data_kernel_f32::prepare_output(int ur_w)
{
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };

    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            xa::ZRegS zreg = zreg_out_s(j, k);
            CGA64::fmov(zreg);
#if 0
            size_t aux_src_offset
                = (size_t)typesize * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j)
                * jcp.ic_block;
            mic_prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                        reg_long_offt));
#endif
        }
    }
}

void jit_sve_conv_bwd_data_kernel_f32::store_output(int ur_w)
{

    auto zreg_tmp = [=](){
        return xa::ZReg(31);
    };
    auto zreg_tmp_s = [=](){
        return xa::ZRegS(31);
    };

    auto zreg_out = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZReg(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };
    auto out_load = [=] (int aux_output_offset){
        int ofs = aux_output_offset;
        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::ldr(zreg_tmp(), xa::ptr(reg_src, static_cast<int32_t>(ofs)));
        }else{
            add_imm(reg_tmp_addr, reg_src, ofs);
            CGA64::ldr(zreg_tmp(), xa::ptr(reg_tmp_addr));
        }
    };

    auto out_str = [=](int j, int k, int aux_output_offset){
        int ofs = aux_output_offset;        
        
        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::str(zreg_out(j, k), xa::ptr(reg_src, static_cast<int32_t>(ofs)));
        }else{ 
            add_imm(reg_tmp_addr, reg_src, ofs); 
            CGA64::str(zreg_out(j, k), xa::ptr(reg_tmp_addr));
        }
    };


    xa::LabelAArch64 no_update_label;

    CGA64::ldr(reg_channel, xa::ptr(param, GET_OFF(channel)));
    CGA64::cmp(reg_channel, 0);
    CGA64::b(xa::EQ, no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;
            out_load(aux_src_offset);
            CGA64::fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s());
        }
    }

    CGA64::L_aarch64(no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;

            out_str(j, k, aux_src_offset);
        }
    }
}

void jit_sve_conv_bwd_data_kernel_f32::compute_loop_fma(
        int ur_w, int l_overflow, int r_overflow)
{
    xa::LabelAArch64 kh_label, kd_label;
    int kw = jcp.kw;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;


    auto zreg_ker = [=](int i_ic) {
        assert(i_ic < 4);
        assert((ker_reg_base_idx + i_ic)<31);
        return xa::ZReg(ker_reg_base_idx + i_ic);
    };
    auto zreg_ker_s = [=](int i_ic) {
        assert(i_ic < 4);
        assert((ker_reg_base_idx + i_ic)<31);
        return xa::ZRegS(ker_reg_base_idx + i_ic);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = (i_ur + i_oc * jcp.ur_w );
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };

    auto bcast_load = [&]( int aux_output_offset, int prev_ofs){
        if( ((aux_output_offset & 0x3) ==0) && 
                (aux_output_offset < LDRWMAX) && 
                    (aux_output_offset >= 0)){
             CGA64::ld1rw(xa::ZRegS(31), reg_p_all_ones,
                    xa::ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        }else{
            int ofs;
            ofs = aux_output_offset - prev_ofs;
            if( ((ofs & 0x3) ==0) && (ofs < LDRWMAX) && (ofs >= 0)){
            
                CGA64::ld1rw(xa::ZRegS(31), reg_p_all_ones,
                    xa::ptr(reg_prev_bcast_addr, static_cast<int32_t>(ofs)));
            }else{
                ofs = aux_output_offset;
                add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs);
    
                CGA64::ld1rw(xa::ZRegS(31), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };
    auto ker_load = [=]( int i, int aux_kernel_offset){
        int ofs = aux_kernel_offset;

        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::ldr(zreg_ker(i), xa::ptr(aux_reg_ker, static_cast<int32_t>(ofs)));
        }else{
            add_imm(reg_tmp_addr, aux_reg_ker, ofs);
            CGA64::ldr(zreg_ker(i), xa::ptr(reg_tmp_addr));
        }

    };



    if (one_of(jcp.ndims, 3, 4)) {
        CGA64::mov(aux_reg_dst, reg_dst);
        CGA64::mov(aux_reg_ker, reg_ker);

        CGA64::mov(aux_reg_dst_prf, reg_dst_prf);
        CGA64::mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        CGA64::ldr(reg_ki, xa::ptr(param , GET_OFF(kd_padding)));
        CGA64::mov(aux_reg_dst_d, reg_dst);
        CGA64::ldr(aux_reg_ker_d, xa::ptr(param, GET_OFF(filt)));
        CGA64::mov(aux_reg_dst_d_prf, reg_dst_prf);
        CGA64::mov(aux_reg_ker_d_prf, reg_ker_prf);

        CGA64::L_aarch64(kd_label);
        CGA64::ldr(reg_kj, xa::ptr(param, GET_OFF(kh_padding)));
    } else {
        CGA64::mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        CGA64::mov(aux_reg_dst, aux_reg_dst_d);
        CGA64::mov(aux_reg_ker, aux_reg_ker_d);
        CGA64::mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        CGA64::mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }
    int prev_ofs = 0;
    CGA64::L_aarch64(kh_label); {
        int step = 0;
        int ker_prfs = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        int aux_kernel_offset = typesize * ((oc + i) * oc_block
                                + ki * ic_block * oc_block);
                        ker_load(i, aux_kernel_offset);
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    int aux_kernel_offset = typesize * ((oc + load_offset)
                            * oc_block + ki * ic_block * oc_block);
                    ker_load(ker_load_reg_idx, aux_kernel_offset);
                }

                bool ker_prf_inserted = false;
                auto zreg_kernel_s = zreg_ker_s(step % ker_pipeline_depth);

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start == nstl::max(0,
                            l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end == ur_w - nstl::max(0,
                            r_overflow - ki * dilate_w));

                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki * dilate_w) % stride_w == 0);
                    int aux_dst_offset = typesize *
                        (((jj + l_pad - ki * dilate_w)
                                / stride_w) * jcp.oc_block + oc);
                    prev_ofs = bcast_load(aux_dst_offset, prev_ofs);
                    CGA64::fmla(zreg_out_s(jj, 0), reg_p_all_ones,
                            zreg_kernel_s, xa::ZRegS(31));

                    
                    int fma_idx = (step * ur_w + jj) / stride_w;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    /*
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (!ker_prf_inserted && ker_prfs < num_ker_loads) {
                            int ker_prf_offset = typesize
                                * ker_prfs * jcp.oc_block;
                            mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_ker_prf, ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_offset
                                    = ic_block * typesize
                                    * ((inp_prf_idx / kw) * kw
                                            + (inp_prf_idx % kw));
                                mic_prefetcht0(EVEX_compress_addr(
                                            aux_reg_dst_prf, inp_prf_offset));
                            }
                        }
                    }
                    */
                }
                step++;
            }
        }

        add_imm(aux_reg_ker, aux_reg_ker, typesize * stride_h * kw * oc_block * ic_block);
        add_imm(aux_reg_dst, aux_reg_dst, -1.0 * typesize * (jcp.dilate_h + 1) * ow * oc_block);
        add_imm(aux_reg_ker_prf, aux_reg_ker_prf, 
                    typesize * stride_h * kw * oc_block * ic_block);
        add_imm(aux_reg_dst_prf, aux_reg_dst_prf, 
                    -1.0 * typesize * (jcp.dilate_h + 1) * ow * oc_block);
        CGA64::sub(reg_kj, reg_kj, 1);
        CGA64::cmp(reg_kj, 0);
        CGA64::b(xa::GT, kh_label); //jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add_imm(aux_reg_dst_d, aux_reg_dst_d,
                -1.0 * typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add_imm(aux_reg_ker_d, aux_reg_ker_d, typesize * jcp.stride_d * jcp.kw * jcp.kh
                * oc_block * ic_block);
        add_imm(aux_reg_dst_d_prf, aux_reg_dst_d_prf,
                -1.0 * typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add_imm(aux_reg_ker_d_prf, aux_reg_ker_d_prf, 
                typesize * jcp.stride_d * jcp.kw * jcp.kh * oc_block * ic_block);

        CGA64::sub(reg_ki, reg_ki, 1);
        CGA64::cmp(reg_ki, 0);
        CGA64::b(xa::GT, kd_label); //jg(kd_label, T_NEAR);
    }

    if (jcp.ndims == 5)
    {
        pop(reg_src);
        pop(reg_src_prf);
    }
}


void jit_sve_conv_bwd_data_kernel_f32::compute_loop_fma_core(
        int ur_w, int l_overflow, int r_overflow)
{
    int kw = jcp.kw;
    int ow = jcp.ow;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    xa::LabelAArch64 kh_label, kd_label;

    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_block;

    auto output_offset = [=](int oi, int oc, int ki) {
        return typesize *
            (((oi + jcp.l_pad - ki * dilate_w) / stride_w) * oc_block + oc);
    };
    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };
    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking){
        int idx = ( i_ic + nb_x_blocking * jcp.ur_w );
        assert(idx < 31);
        return xa::ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc){
        int idx = (i_ur + i_oc * jcp.ur_w );
        assert(idx < ker_reg_base_idx);
        return xa::ZRegS(idx);
    };
    auto zreg_wei = [=](){
        return xa::ZReg(31);
    };
    auto zreg_wei_s = [=](){
        return xa::ZRegS(31);
    };


    auto bcast_load = [&](int jj, int nb_oc_block, int aux_output_offset, int prev_ofs, int jj_end){
        if( ((aux_output_offset & 0x3) ==0) && 
                (aux_output_offset < LDRWMAX) && 
                    (aux_output_offset >= 0)){
            CGA64::ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                    xa::ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        }else{
            if( (prev_ofs > -1) && 
                ((aux_output_offset - prev_ofs)>0) &&
                ((aux_output_offset - prev_ofs) < LDRWMAX) && 
                (((aux_output_offset - prev_ofs)& 0x3) ==0)){
                
                CGA64::ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                        xa::ptr(reg_prev_bcast_addr, static_cast<int32_t>(aux_output_offset - prev_ofs)));
    
            }else{
                int ofs;
                if((prev_ofs > -1) && ((aux_output_offset - prev_ofs)>0)){
                    ofs = aux_output_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs);
    
                }else{
                    ofs = aux_output_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs);
                }
    
                CGA64::ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr));
                prev_ofs = aux_output_offset;
            }
        }
        return prev_ofs;
    };

    auto bcast_load_30 = [&](int jj, int nb_oc_block, int aux_output_offset, int prev_ofs, int jj_end){
        if( ((aux_output_offset & 0x3) ==0) && 
                (aux_output_offset < LDRWMAX) && 
                    (aux_output_offset >= 0)){
            CGA64::ld1rw(xa::ZRegS(30), reg_p_all_ones,
                    xa::ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        }else{
            if( (prev_ofs > -1) && 
                ((aux_output_offset - prev_ofs)>0) &&
                ((aux_output_offset - prev_ofs) < LDRWMAX) && 
                (((aux_output_offset - prev_ofs)& 0x3) ==0)){
                
                CGA64::ld1rw(xa::ZRegS(30), reg_p_all_ones,
                        xa::ptr(reg_prev_bcast_addr, static_cast<int32_t>(aux_output_offset - prev_ofs)));
    
            }else{
                int ofs;
                if((prev_ofs > -1) && ((aux_output_offset - prev_ofs)>0)){
                    ofs = aux_output_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs);
    
                }else{
                    ofs = aux_output_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs);
                }
    
                CGA64::ld1rw(xa::ZRegS(30), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr));
                prev_ofs = aux_output_offset;
            }
        }
        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset){
        int ofs = aux_kernel_offset;

        if( ((ofs>>6) < LDRMAX) && 
                ((ofs>>6) >= (-1.0* LDRMAX)) &&
                ((ofs&0x3f) == 0)){
            ofs = ofs >>6;
            CGA64::ldr(zreg_wei(), xa::ptr(aux_reg_ker, static_cast<int32_t>(ofs)));
        }else{
            add_imm(reg_tmp_addr, aux_reg_ker, ofs);
            CGA64::ldr(zreg_wei(), xa::ptr(reg_tmp_addr));
        }

    };

    if (one_of(jcp.ndims, 3, 4)) {
        CGA64::mov(aux_reg_dst, reg_dst);
        CGA64::mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        CGA64::ldr(reg_ki, xa::ptr(param, GET_OFF(kd_padding)));
        CGA64::mov(aux_reg_dst_d, reg_dst);
        CGA64::ldr(aux_reg_ker_d, xa::ptr(param, GET_OFF(filt)));

        CGA64::L_aarch64(kd_label);
        CGA64::ldr(reg_kj, xa::ptr(param, GET_OFF(kh_padding)));
    } else {
        CGA64::mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        CGA64::mov(aux_reg_dst, aux_reg_dst_d);
        CGA64::mov(aux_reg_ker, aux_reg_ker_d);
    }

    CGA64::L_aarch64(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) { // kernel width
            int prev_ofs = -1;

            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (stride_w == 1) {
                    for (int jj = jj_start; jj < jj_end; jj += 1) {
                        int aux_output_offset = output_offset(jj, oc, ki);
                        prev_ofs = bcast_load(jj, nb_ic_block, aux_output_offset, prev_ofs, jj_end);
                    }
                }
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    int aux_kernel_offset = kernel_offset(ii, oc, ki);
                    if (jj_end - jj_start > 0){
                        wei_load(aux_kernel_offset);
                    }
                    for (int jj = jj_start; jj < jj_end; jj += stride_w){
                        if(stride_w == 1){
                            CGA64::fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                        zreg_inp_s(jj, nb_ic_block), zreg_wei_s());
                        }else{
                            int aux_output_offset = output_offset(jj, oc, ki);
                            prev_ofs = bcast_load_30(jj, nb_ic_block, aux_output_offset, prev_ofs, jj_end);

                            CGA64::fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                        xa::ZRegS(30), zreg_wei_s());

                        }
                    }

                }
            }
        }
        add_imm(aux_reg_ker, aux_reg_ker, shift_ker_ptr);
        assert(shift_dst_ptr < 4095);
        assert(shift_dst_ptr > 0);
        CGA64::sub(aux_reg_dst, aux_reg_dst, shift_dst_ptr);
        //dec(reg_kj);
        CGA64::sub(reg_kj, reg_kj, 1);
        CGA64::cmp(reg_kj, 0);
        CGA64::b(xa::GT, kh_label);
    }

    if (jcp.ndims == 5) {
        assert((typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block) < 4095);
        CGA64::sub(aux_reg_dst_d, aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add_imm(aux_reg_ker_d, aux_reg_ker_d,  typesize * jcp.kw * jcp.kh * oc_block * ic_block);

        //dec(reg_ki);
        CGA64::sub(reg_ki, reg_ki, 1);
        CGA64::cmp(reg_ki, 0);
        CGA64::b(xa::GT, kd_label);

        pop(reg_src);
        pop(reg_src_prf);
    }
}

inline void jit_sve_conv_bwd_data_kernel_f32::compute_loop(
        int ur_w, int l_overflow, int r_overflow)
{
    if (jcp.ndims == 5){
        push(reg_oi);
    }

    prepare_output(ur_w);

    xa::LabelAArch64 skip_compute_loop;
    if (jcp.ndims == 5) {
        CGA64::ldr(reg_kj, xa::ptr(param, GET_OFF(kd_padding)));
        CGA64::cmp(reg_kj, 0);
        CGA64::b(xa::LE, skip_compute_loop);
    }
    CGA64::ldr(reg_kj, xa::ptr(param, GET_OFF(kh_padding)));
    CGA64::cmp(reg_kj, 0);
    CGA64::b(xa::LE, skip_compute_loop);

    if (jcp.ver == ver_fma)
        if ( jcp.nb_ic_blocking == 1)
            compute_loop_fma(ur_w, l_overflow, r_overflow);
        else
            compute_loop_fma_core(ur_w, l_overflow, r_overflow);
    else
        assert("!unknown convolution version");

    CGA64::L_aarch64(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) {
        pop(reg_oi);
    }
}

void jit_sve_conv_bwd_data_kernel_f32::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();
    CGA64::ptrue( reg_p_all_ones.b );
    /* Address */
    CGA64::ldr(reg_src, xa::ptr(param, GET_OFF(src)));
    CGA64::ldr(reg_dst, xa::ptr(param, GET_OFF(dst)));
    CGA64::ldr(reg_ker, xa::ptr(param, GET_OFF(filt)));
 
    CGA64::ldr(reg_kh, xa::ptr(param, GET_OFF(kh_padding)));
    CGA64::ldr(reg_src_prf, xa::ptr(param, GET_OFF(src_prf)));
    CGA64::ldr(reg_dst_prf, xa::ptr(param, GET_OFF(dst_prf)));
    CGA64::ldr(reg_ker_prf, xa::ptr(param, GET_OFF(filt_prf)));

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(
            0, ((kw - 1) * dilate_w - jcp.r_pad - ur_w_tail) / stride_w);

    int n_oi = iw / ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) { // Case: workload can run in a single compute_loop call
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add_imm(reg_src, reg_src, src_shift);
        add_imm(reg_dst, reg_dst, dst_shift);
        add_imm(reg_src_prf, reg_src_prf, src_shift);
        add_imm(reg_dst_prf, reg_dst_prf, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        CGA64::mov(reg_oi, 0);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add_imm(reg_src, reg_src, src_shift);
            add_imm(reg_dst, reg_dst, dst_shift);
            add_imm(reg_src_prf, reg_src_prf, src_shift);
            add_imm(reg_dst_prf, reg_dst_prf, dst_shift);

            add_imm(reg_oi, reg_oi, 1);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow > 0 && n_oi > 1)) {
            xa::LabelAArch64 ow_loop_label;
            CGA64::L_aarch64(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add_imm(reg_src, reg_src, src_shift);
                add_imm(reg_dst, reg_dst, dst_shift);
                add_imm(reg_src_prf, reg_src_prf, src_shift);
                add_imm(reg_dst_prf, reg_dst_prf, dst_shift);

                add_imm( reg_oi, reg_oi, 1);
                CGA64::cmp(reg_oi, n_oi);
                CGA64::b(xa::LT, ow_loop_label);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add_imm(reg_src, reg_src, src_shift);
            add_imm(reg_dst, reg_dst, dst_shift);
            add_imm(reg_src_prf, reg_src_prf, src_shift);
            add_imm(reg_dst_prf, reg_dst_prf, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

status_t jit_sve_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(sve)) return status::unimplemented;

    jcp = zero<decltype(jcp)>();

    jcp.simd_w = cpu_isa_traits<sve>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    /* XXX BUGBUGBUG: current workaround to support negative padding: use the
     * 'stride-complement' of padding instead. */
    if (jcp.kh == 1 && jcp.b_pad < 0) jcp.b_pad = jcp.stride_h + jcp.b_pad;
    if (jcp.kd == 1 && jcp.back_pad < 0)
        jcp.back_pad = jcp.stride_d + jcp.back_pad;

    jcp.aligned_threads = 0;

    jcp.is_1stconv = false;

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && diff_src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto src_format = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_format = with_groups
        ? pick(ndims - 3, gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
        : pick(ndims - 3, OIw16o16i, OIhw16o16i, OIdhw16o16i);
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0
        && diff_src_d.format() == src_format
        && diff_dst_d.format() == src_format;
    if (!args_ok)
        return status::unimplemented;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    int regs = 26;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.r_pad - jcp.iw % jcp.ur_w)
                    / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (mayiuse(sve)
         && diff_dst_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && diff_src_d.data_type() == data_type::f32) {
        if (weights_d.format() != wei_format)
            return status::unimplemented;
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
            
    } else {
        return status::unimplemented;
    }
    if (!utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && jcp.ver != ver_fma)
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    
    jcp.loop_order = loop_gnc;

    // Heuristic to optimize code size on KNX
    bool large_code_size = (jcp.ur_w != jcp.ow)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow > 0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(sve)) {
        int try_nb_ic_blocking = 2;
        unsigned int ker_inp_size = typesize * jcp.iw * jcp.ic_block
            * try_nb_ic_blocking * jcp.kh;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_ic_blocking;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;

        if (!(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
            || (jcp.kw < 5 && ((jcp.iw <= 5 || (jcp.iw > 8 && jcp.iw <= 13)))))
                || jcp.stride_h > 1 || jcp.stride_d > 1) {
            jcp.kernel_kind = expl_bcast;
            jcp.ur_w = nstl::min(jcp.iw, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && jcp.ow > 8)) && jcp.stride_h == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 30 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
            
        } else 
        {

            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            if(jcp.stride_w > 1){
                jcp.ur_w = 30 / (jcp.nb_ic_blocking + 1);
            }else{
                jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            }
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.r_pad - jcp.ur_w_tail)
                    / jcp.stride_w);
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;
    args_ok = true
        && jcp.ic <= diff_src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (l_overflow > 0) + (r_overflow1 > 0);
        const float max_instruction_size = 15;
        float ur_fac
                = (float)jcp.kw * jcp.oc_block * jcp.nb_ic_blocking * jcp.ur_w;
        float code_size = mult * ur_fac * max_instruction_size;
        if (code_size > max_code_size) return status::unimplemented;
    }

    return status::success;
}

void jit_sve_conv_bwd_data_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

// Initialize static data members
const int jit_sve_conv_bwd_weights_kernel_f32::max_ur_w = 28;
const int jit_sve_conv_bwd_weights_kernel_f32::min_oh_reduce = 9;

void jit_sve_conv_bwd_weights_kernel_f32::od_step_comeback_pointers()
{
    xa::LabelAArch64 kd_comeback_label;

    /* 'depth' loop count bound by 'kd_work_size' */
    CGA64::mov(kj, reg_kd_count);
    CGA64::L_aarch64(kd_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni))
            ? jcp.tr_iw : jcp.iw;
        add_imm(reg_input, reg_input,
                -1 * jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mult);
        add_imm(reg_kernel, reg_kernel,
                -1 * jcp.typesize_out * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block);
        CGA64::sub(kj, kj, 1);
        CGA64::cmp(kj, 0);
        CGA64::b(xa::GT, kd_comeback_label);
    }
}

void jit_sve_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers()
{
    xa::LabelAArch64 kh_comeback_label, kd_comeback_label;
    CGA64::mov(kj, reg_kh);
    CGA64::L_aarch64(kh_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni))
            ? jcp.tr_iw : jcp.iw;
        add_imm(reg_input, reg_input,
                -1 * jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mult);
        add_imm(reg_kernel, reg_kernel,
                -1 * jcp.typesize_out * jcp.kw * jcp.ic_block * jcp.oc_block);
        CGA64::sub(kj, kj, 1);
        CGA64::cmp(kj, 0);
        CGA64::b(xa::GT, kh_comeback_label);
    }
}

void jit_sve_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{

    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            add_imm(reg_add_tmp, reg_kernel, typesize * (i_kw * ic_block + i_ic)
                     * jcp.oc_block + kernel_offset);
            CGA64::ldr(xa::ZReg(i_kw * ic_block_step + i_ic), xa::ptr(reg_add_tmp));
                //vmovups(Zmm(i_kw * ic_block_step + i_ic),
                //  EVEX_compress_addr(reg_kernel, typesize * (i_kw * ic_block
                //  + i_ic) * jcp.oc_block + kernel_offset));
        }
    }

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        if (i_ur == 0) {
            add_imm(reg_add_tmp, reg_output, typesize * (i_ur + 0) * oc_block + output_offset);
            assert(kw * ic_block_step + (i_ur + 0) % 4 < 31);
            CGA64::ldr(xa::ZReg(kw * ic_block_step + (i_ur + 0) % 4), xa::ptr(reg_add_tmp));
                //vmovups(Zmm(kw * ic_block_step + (i_ur + 0) % 4),
                //    EVEX_compress_addr(reg_output, typesize * (i_ur + 0)
                //    * oc_block + output_offset));
            if (ur_w > 1) {
                add_imm(reg_add_tmp, reg_output, typesize * (i_ur + 1) * oc_block + output_offset);
                assert(kw * ic_block_step + (i_ur + 1) % 4 < 31);
                CGA64::ldr(xa::ZReg(kw * ic_block_step + (i_ur + 1) % 4), xa::ptr(reg_add_tmp));
                    //vmovups(Zmm(kw * ic_block_step + (i_ur + 1) % 4),
                    //EVEX_compress_addr(reg_output, typesize * (i_ur + 1) * oc_block
                    //+ output_offset));
            }
            if (ur_w > 2) {
                add_imm(reg_add_tmp, reg_output, typesize * (i_ur + 2) * oc_block + output_offset);
                assert(kw * ic_block_step + (i_ur + 2) % 4 < 31);
                CGA64::ldr(xa::ZReg(kw * ic_block_step + (i_ur + 2) % 4), xa::ptr(reg_add_tmp));
                    //vmovups(Zmm(kw * ic_block_step + (i_ur + 2) % 4),
                    //EVEX_compress_addr(reg_output, typesize * (i_ur + 2) * oc_block
                    //+ output_offset));
            }
            if (ur_w > 3) {
                add_imm(reg_add_tmp, reg_output, typesize * (i_ur + 3) * oc_block + output_offset);
                assert(kw * ic_block_step + (i_ur + 3) % 4 < 31);
                CGA64::ldr(xa::ZReg(kw * ic_block_step + (i_ur + 3) % 4), xa::ptr(reg_add_tmp));
                    //vmovups(Zmm(kw * ic_block_step + (i_ur + 3) % 4),
                    //EVEX_compress_addr(reg_output, typesize * (i_ur + 3) * oc_block
                    //+ output_offset));
            }
        } else if (i_ur + 3 < ur_w) {
            add_imm(reg_add_tmp, reg_output, typesize * (i_ur + 3) * oc_block + output_offset);
            assert(kw * ic_block_step + (i_ur + 3) % 4 < 31);
            CGA64::ldr(xa::ZReg(kw * ic_block_step + (i_ur + 3) % 4), xa::ptr(reg_add_tmp));
                // vmovups(Zmm(kw * ic_block_step + (i_ur + 3) % 4),
                //     EVEX_compress_addr(reg_output, typesize * (i_ur + 3) * oc_block
                //     + output_offset));
        }

        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur * jcp.stride_w + i_kw * (jcp.dilate_w + 1);
            if (i_iw - pad_l < 0 || i_iw > (ur_w - 1) * jcp.stride_w +
                    (kw - 1) * (jcp.dilate_w + 1) - pad_r) continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                const size_t i_offset = (size_t)input_offset
                    + (size_t)typesize * (jcp.ver == ver_4fma
                            ? (i_iw - pad_l + i_ic * jcp.tr_iw)
                            : (jcp.is_1stconv
                                ? (i_iw - pad_l) + (size_t)i_ic
                                    * ((size_t)jcp.ih*jcp.iw*jcp.id)
                                : (i_iw - pad_l) * ic_block + i_ic));
                assert((i_kw * ic_block_step + i_ic) < 31);
                assert((kw * ic_block_step + (i_ur % 4)) < 31);
                assert(i_offset < (1<<31));
                add_imm(reg_add_tmp, reg_input, i_offset);
                ld1rw(zreg_idata, reg_p_all_ones, xa::ptr(reg_add_tmp));
                CGA64::fmla(xa::ZRegS(i_kw * ic_block_step + i_ic), reg_p_all_ones,
                            xa::ZRegS(kw * ic_block_step + i_ur % 4),
                            xa::ZRegS(zreg_idata));
                    //vfmadd231ps(Zmm(i_kw * ic_block_step + i_ic),
                    //    Zmm(kw * ic_block_step + i_ur % 4),
                    //    EVEX_compress_addr_safe(reg_input, i_offset, reg_long_offt, true));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            add_imm(reg_add_tmp, reg_kernel, typesize * (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset);
            CGA64::str(xa::ZReg(i_kw * ic_block_step + i_ic), xa::ptr(reg_add_tmp));
                //vmovups(EVEX_compress_addr(reg_kernel, typesize
                //    * (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset),
                //    Zmm(i_kw * ic_block_step + i_ic));
        }
    }
}

void jit_sve_conv_bwd_weights_kernel_f32
    ::compute_oh_step_unroll_ow_icblock(
    int ic_block_step, int max_ur_w)
{
    UNUSED(max_ur_w);

    xa::LabelAArch64 kh_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int iw = (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni))
        ? jcp.tr_iw : jcp.iw;
    int ow = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? jcp.tr_ow : jcp.ow;

    int r_pad = nstl::max(0, (ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        CGA64::L_aarch64(kd_label);
        CGA64::mov(reg_input, aux_reg_input);
        CGA64::mov(reg_kernel, aux_reg_kernel);
    }

    CGA64::mov(kj, reg_kh);
    CGA64::L_aarch64(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = jcp.typesize_in
                * (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)
                   ? i_b_ic * iw : i_b_ic);
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                input_offset, jcp.typesize_out * i_b_ic * jcp.oc_block, 0,
                i_b_ic + ic_block_step >= jcp.ic_block);
        }
        add_imm(reg_input, reg_input,
                jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        add_imm(reg_kernel, reg_kernel,
                jcp.typesize_out * jcp.kw * ic_block * oc_block);
        CGA64::sub(kj, kj, 1);
        CGA64::cmp(kj, 0);
        CGA64::b(xa::GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mul);
        add_imm(aux_reg_kernel, aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        CGA64::sub(ki, ki, 1);
        CGA64::cmp(ki, 0);
        CGA64::b(xa::GT, kd_label);
    }
}

void jit_sve_conv_bwd_weights_kernel_f32
    ::compute_oh_step_unroll_ow(
    int ic_block_step, int max_ur_w)
{
    xa::LabelAArch64 kh_label, ic_block_label, kd_label;

    UNUSED(max_ur_w);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int ow = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? jcp.tr_ow : jcp.ow;

    int r_pad = nstl::max(0,
        (ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        CGA64::L_aarch64(kd_label);
        CGA64::mov(reg_input, aux_reg_input);
        CGA64::mov(reg_kernel, aux_reg_kernel);
    }

    CGA64::mov(kj, reg_kh);
    CGA64::L_aarch64(kh_label);
    {
        CGA64::mov(b_ic, 0);
        CGA64::L_aarch64(ic_block_label); {
            compute_ic_block_step(ow, l_pad, r_pad, ic_block_step,
                0, 0, 0);
            size_t inp_icblk_stride = jcp.is_1stconv
                ? (size_t)jcp.ih * jcp.iw * jcp.id
                : (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)
                   ? jcp.tr_iw : 1);
            size_t input_offset
                = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            add_imm(reg_input, reg_input, input_offset);
                //safe_add(reg_input, input_offset, reg_long_offt);
            add_imm(reg_kernel, reg_kernel, jcp.typesize_out * ic_block_step * oc_block);
            add_imm(b_ic, b_ic, ic_block_step);
            assert(jcp.ic_block < ADDMAX);
            CGA64::cmp(b_ic, jcp.ic_block);
            CGA64::b(xa::LT, ic_block_label);
        }

        if (jcp.is_1stconv) {
            size_t input_offset
                = (size_t)jcp.typesize_in * jcp.id * jcp.ih * jcp.iw * ic_block;
            add_imm(reg_input, reg_input, (long long int)(-1) * input_offset  );
                //safe_sub(reg_input, input_offset, reg_long_offt);
            add_imm(reg_input, reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw);
        } else if (!utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)) {
            add_imm(reg_input, reg_input, jcp.typesize_in
                    * ((jcp.dilate_h + 1) * jcp.iw - 1) * ic_block);
        }
        add_imm(reg_kernel, reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        CGA64::sub(kj, kj, 1);
        CGA64::cmp(kj, 0);
        CGA64::b(xa::GT, kh_label);
    }
    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input, jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih
                * jcp.iw * (jcp.is_1stconv ? 1 : ic_block));
        add_imm(aux_reg_kernel, aux_reg_kernel, jcp.typesize_out * jcp.kh * jcp.kw * ic_block
                * oc_block);
        CGA64::sub(ki, ki, 1);
        CGA64::cmp(ki, 0);
        CGA64::b(xa::GT, kd_label);
    }
}

void jit_sve_conv_bwd_weights_kernel_f32
    ::compute_oh_step_common(
    int ic_block_step, int max_ur_w)
{
    xa::LabelAArch64 kh_label, ic_block_label, ow_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int ow = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? jcp.tr_ow : jcp.ow;
    int r_pad = nstl::max(0, (ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    int l_pad = (jcp.ver == ver_4fma || jcp.ver == ver_4vnni
                 || jcp.ver == ver_vnni) ? 0 : jcp.l_pad;

    int ur_w = nstl::min(ow, max_ur_w);
    int ur_w_trips = ow / ur_w;
    int ur_w_tail = ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0)
        || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    int inp_mult = (jcp.is_1stconv ||
        utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)) ? 1 : ic_block;
    int input_comeback = (ur_w_trips * ur_w * jcp.stride_w - l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    if (jcp.ndims == 5) {
        CGA64::L_aarch64(kd_label);
        CGA64::mov(reg_input, aux_reg_input);
        CGA64::mov(reg_kernel, aux_reg_kernel);
    }

    CGA64::mov(kj, reg_kh);
    CGA64::L_aarch64(kh_label); {
        CGA64::mov(b_ic, 0);
        CGA64::L_aarch64(ic_block_label); {
            if (l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add_imm(reg_input, reg_input, jcp.typesize_in * (ur_w * jcp.stride_w - l_pad)
                        * inp_mult);
                add_imm(reg_output, reg_output, jcp.typesize_in * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                CGA64::mov(reg_ur_w_trips, 0);
                CGA64::L_aarch64(ow_block_label); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add_imm(reg_input, reg_input, jcp.typesize_in * ur_w * jcp.stride_w
                            * inp_mult);
                    add_imm(reg_output, reg_output, jcp.typesize_in * ur_w * oc_block);

                    add_imm(reg_ur_w_trips, reg_ur_w_trips, 1);
                    CGA64::cmp(reg_ur_w_trips, ur_w_trips);
                    CGA64::b(xa::LT, ow_block_label);
                }
            }

            if (ur_w_tail > 0) compute_ic_block_step(ur_w_tail, 0, r_pad,
                ic_block_step, 0, 0, 0);

            add_imm(reg_input, reg_input, -1 * jcp.typesize_in * input_comeback);
            add_imm(reg_output, reg_output, -1 * jcp.typesize_in * output_comeback);
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.iw * jcp.id
                : (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)
                   ? jcp.tr_iw : 1);
            size_t input_offset
                = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            add_imm(reg_input, reg_input, input_offset);
                //safe_add(reg_input, input_offset, reg_long_offt);
            add_imm(reg_kernel, reg_kernel, jcp.typesize_out * ic_block_step * oc_block);

            add_imm(b_ic, b_ic, ic_block_step);
            CGA64::cmp(b_ic, jcp.ic_block);
            CGA64::b(xa::LT, ic_block_label);
        }
        if (jcp.is_1stconv) {
            size_t input_offset
                = (size_t)jcp.typesize_in * jcp.id * jcp.ih * jcp.iw * ic_block;
            add_imm(reg_input, reg_input, (long long int)(-1) * input_offset);
                //safe_sub(reg_input, input_offset, reg_long_offt);
            add_imm(reg_input, reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw);
        } else if (!utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)) {
            add_imm(reg_input, reg_input,
                    jcp.typesize_in * ((jcp.dilate_h + 1 ) * jcp.iw - 1) * ic_block);
        }
        add_imm(reg_kernel, reg_kernel,
                jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        CGA64::sub(kj, kj, 1);
        CGA64::cmp(kj, 0);
        CGA64::b(xa::GT, kh_label);
    }
    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input, jcp.typesize_in * (jcp.dilate_d + 1)
                * jcp.ih * jcp.iw * (jcp.is_1stconv ? 1 : ic_block));
        add_imm(aux_reg_kernel, aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        CGA64::sub(ki, ki, 1);
        CGA64::cmp(ki, 0);
        CGA64::b(xa::GT, kd_label);
    }
}

void jit_sve_conv_bwd_weights_kernel_f32
    ::compute_oh_step_disp()
{
    int ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw <= 6 ? 4 : 2);
    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step
            = (jcp.kw * jcp.ic_block <= 26 && !large_code) ? jcp.ic_block : 1;
    }

    bool too_large_to_unroll
        = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
        && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_input = r12. The following order of
         * 'movs' must be guaranteed. */
        CGA64::mov(ki, reg_kd_count);
        push(reg_kd_count);
        CGA64::mov(aux_reg_input, reg_input);
        CGA64::mov(aux_reg_kernel, reg_kernel);
    }

    int ow = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? jcp.tr_ow : jcp.ow;
    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        CGA64::mov(reg_input, aux_reg_input);
        CGA64::mov(reg_kernel, aux_reg_kernel);
        pop(reg_kd_count);
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_sve_conv_bwd_weights_kernel_f32::maybe_zero_kernel()
{
    xa::LabelAArch64 skip_zeroing, zeroing_loop;

    CGA64::ldr(reg_tmp, xa::ptr(param, GET_OFF(channel)));
    CGA64::cmp(reg_tmp, 0);
    CGA64::b(xa::EQ, skip_zeroing);

    xa::ZRegS zero = xa::ZRegS(0);
    CGA64::eor(zero, reg_p_all_ones, zero); //vpxord(zero, zero, zero);
    CGA64::mov(reg_tmp, 0);
    CGA64::L_aarch64(zeroing_loop); {
        assert(jcp.oc_block * jcp.typesize_out
            == cpu_isa_traits<sve>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
            CGA64::add(reg_add_tmp, reg_kernel, reg_tmp);
            add_imm(reg_add_tmp, reg_add_tmp, ic1 * jcp.oc_block * jcp.typesize_out);
            CGA64::str(xa::ZReg(0), xa::ptr(reg_add_tmp));
                //vmovups(ptr[reg_kernel + reg_tmp + ic1 * jcp.oc_block
                //    * jcp.typesize_out], zero);
        }
        add_imm(reg_tmp, reg_tmp, jcp.ic_block * jcp.oc_block * jcp.typesize_out);
        mov_imm(reg_add_tmp, jcp.ic_block * jcp.oc_block * jcp.kw * jcp.kh * jcp.kd * jcp.typesize_out);
        CGA64::cmp(reg_tmp, reg_add_tmp);
        CGA64::b(xa::NE, zeroing_loop);
    }

    CGA64::L_aarch64(skip_zeroing);
}

void jit_sve_conv_bwd_weights_kernel_f32::bias_kernel_2d() {
    assert(jcp.ndims == 4); // only supports 2d
    xa::LabelAArch64 skip_bias, bias_loop;

    CGA64::ldr(reg_tmp, xa::ptr(param, GET_OFF(flags)));
    CGA64::tst(reg_tmp, reg_tmp);
    CGA64::b(xa::NE, skip_bias);

    CGA64::ldr(xa::ZReg(0), xa::ptr(reg_bias)); //vmovups(Zmm(0), ptr[reg_bias]);

    mov_imm(reg_oi, jcp.ow);
    CGA64::mov(reg_tmp, 0);
    CGA64::L_aarch64(bias_loop);
    {
        CGA64::add(reg_add_tmp, reg_output, reg_tmp);
        CGA64::ldr(xa::ZReg(1), xa::ptr(reg_add_tmp));
            //vmovups(Zmm(1), ptr[reg_output + reg_tmp]);
        //CGA64::add(xa::ZRegS(0), reg_p_all_ones, xa::ZRegS(1));
        CGA64::fadd(xa::ZRegS(0), xa::ZRegS(0), xa::ZRegS(1));
            //vaddps(Zmm(0), Zmm(0), Zmm(1));
        add_imm(reg_tmp, reg_tmp, jcp.typesize_out * jcp.oc_block);
        CGA64::sub(reg_oi, reg_oi, 1);
        CGA64::cmp(reg_oi, 0);
        CGA64::b(xa::GT, bias_loop);
    }
    CGA64::str(xa::ZReg(0), xa::ptr(reg_bias)); //vmovups(ptr[reg_bias], Zmm(0));

    CGA64::L_aarch64(skip_bias);
}

void jit_sve_conv_bwd_weights_kernel_f32::bias_kernel_3d() {
    assert(jcp.ndims == 5); // only supports 3d
    xa::LabelAArch64 skip_bias, bias_loop, skip_load_bias;

    CGA64::ldr(reg_tmp, xa::ptr(param, GET_OFF(flags)));
    CGA64::tst(reg_tmp, reg_tmp);
    CGA64::b(xa::NE, skip_bias);

    CGA64::ldr(reg_bias, xa::ptr(param, GET_OFF(bias)));
    CGA64::ldr(reg_output, xa::ptr(param, GET_OFF(dst)));
    CGA64::eor(xa::ZRegS(1), reg_p_all_ones, xa::ZRegS(1)); //vpxord(Zmm(1), Zmm(1), Zmm(1));

    CGA64::ldr(reg_tmp, xa::ptr(param, GET_OFF(channel)));
    CGA64::cmp(reg_tmp, 0);
    CGA64::b(xa::NE, skip_load_bias);
    CGA64::ldr(xa::ZReg(1), xa::ptr(reg_bias)); //vmovups(Zmm(1), ptr[reg_bias]);

    CGA64::L_aarch64(skip_load_bias);

    CGA64::ldr(reg_oi, xa::ptr(param, GET_OFF(os_index_end)));
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(os_index_begin)));
    CGA64::sub(reg_oi, reg_oi, reg_tmp_imm); //sub(reg_oi, ptr[param + GET_OFF(os_index_begin)]);
    CGA64::cmp(reg_oi, 0);
    CGA64::b(xa::LE, skip_bias); // no iterations along depth dimension

    mov_imm(reg_tmp, jcp.oc_block * jcp.ow * jcp.oh * jcp.typesize_out);
    CGA64::mul(reg_oi, reg_oi, reg_tmp);

    CGA64::mov(reg_tmp, 0);
    CGA64::L_aarch64(bias_loop); {
        CGA64::ldr(xa::ZReg(0), xa::ptr(reg_tmp)); //vmovups(Zmm(0), ptr[reg_output + reg_tmp]);
        CGA64::add(xa::ZRegS(1), reg_p_all_ones, xa::ZRegS(0)); //vaddps(Zmm(1), Zmm(1), Zmm(0));
        add_imm(reg_tmp, reg_tmp, jcp.oc_block * jcp.typesize_out);
        CGA64::cmp(reg_tmp, reg_oi);
        CGA64::b(xa::LT, bias_loop);
    }
    CGA64::str(xa::ZReg(1), xa::ptr(reg_bias)); //vmovups(ptr[reg_bias], Zmm(1));

    CGA64::L_aarch64(skip_bias);
}

void jit_sve_conv_bwd_weights_kernel_f32
    ::compute_oh_loop_common()
{
    assert(one_of(jcp.harness, harness_mb_reduction, harness_3d_reduction));
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni) ? jcp.tr_iw
        : jcp.iw;
    xa::LabelAArch64 oh_label, oh_label_end, oh_tpad_label, oh_tpad_tail_label,
            oh_bpad_label, oh_bpad_label_end, oh_dilate_label_shift,
            oh_dilate_label_noshift, oh_dilate_label_end;

    int ow = jcp.ow;

    CGA64::mov(reg_kh, jcp.kh);
    CGA64::mov(reg_ih_count, 0);
    CGA64::mov(reg_oj, 0);
    /* Compute 'top' edge */
    if (t_pad > 0) {
        const int kh_range = 1 + (jcp.kh - 1) * dilate_h;
        const int overflow
            = nstl::max(0, jcp.kh - div_up(t_pad + jcp.ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_inp_ker_overlap = jcp.kh - overflow - underflow;
        mov_imm(reg_kh, initial_inp_ker_overlap); //mov(reg_kh, initial_inp_ker_overlap);
        add_imm(reg_kernel, reg_kernel, jcp.typesize_out * underflow * jcp.kw
                * jcp.ic_block * jcp.oc_block);
        // generate loop to process kernel while it remains within t_pad + ih
        if (kh_range < t_pad + jcp.ih) {
            if (is_dilated) {
                const int tail = t_pad % dilate_h;
                const int shift = tail == 0 ? 0 : dilate_h - tail;
                mov_imm(reg_tmp, shift); //mov(reg_tmp, shift);
                if (tail != 0)
                    add_imm(reg_input, reg_input,
                            jcp.typesize_in * shift * iw * inp_mult);
            }
            CGA64::L_aarch64(oh_tpad_label); {
                CGA64::cmp(reg_oj, jcp.oh);
                CGA64::b(xa::GE, oh_label_end);

                compute_oh_step_disp();
                add_imm(reg_output, reg_output, jcp.typesize_in * ow * jcp.oc_block);
                if (is_dilated) {
                    add_imm(reg_tmp, reg_tmp, 1);
                    CGA64::cmp(reg_tmp, dilate_h);
                    CGA64::b(xa::LT, oh_dilate_label_shift);
                    // unshift input as new kernel element enters
                    add_imm(reg_input, reg_input,
                            -1 * jcp.typesize_in * (dilate_h - 1) * iw * inp_mult);
                    CGA64::mov(reg_tmp, 0);
                }
                // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
                add_imm(reg_kernel, reg_kernel, -1 * jcp.typesize_out * stride_h * jcp.kw
                        * jcp.ic_block * jcp.oc_block);
                add_imm(reg_kh, reg_kh, stride_h);
                if (is_dilated) {
                    CGA64::b(oh_dilate_label_noshift);
                    CGA64::L_aarch64(oh_dilate_label_shift);
                    // shift input as old kernel element progresses
                    add_imm(reg_input, reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
                    CGA64::L_aarch64(oh_dilate_label_noshift);
                }
                add_imm(reg_oj, reg_oj, 1);
                add_imm(reg_ih_count, reg_ih_count, stride_h);

                // final number of kernel elements that overlap with input
                const int final_inp_ker_overlap
                    = nstl::min(jcp.kh, div_up(jcp.ih, dilate_h));
                CGA64::cmp(reg_kh, final_inp_ker_overlap);
                CGA64::b(xa::LT, oh_tpad_label);
            }
        }
        // need second loop to process kernel if it is larger than the input
        // (does not apply to dilations as they must have unit stride)
        if (kh_range >= jcp.ih + (t_pad % stride_h == 0 ? stride_h :
                                                        t_pad % stride_h)) {
            assert(!is_dilated);
            CGA64::mov(reg_kh, jcp.ih);
            CGA64::L_aarch64(oh_tpad_tail_label); {
                CGA64::cmp(reg_oj, jcp.oh);
                CGA64::b(xa::GE, oh_label_end);

                compute_oh_step_disp();
                add_imm(reg_output, reg_output, jcp.typesize_in * ow * jcp.oc_block);
                add_imm(reg_kernel, reg_kernel, -1 * jcp.typesize_out * stride_h * jcp.kw
                        * jcp.ic_block * jcp.oc_block);

                add_imm(reg_oj, reg_oj, 1);
                add_imm(reg_ih_count, reg_ih_count, stride_h);

                CGA64::cmp(reg_ih_count, nstl::min(t_pad, jcp.oh * stride_h));
                CGA64::b(xa::LT, oh_tpad_tail_label);
            }
        }
        // correct any excess shifts to kernel and input
        // (does not apply to dilations as they must have unit stride,
        //  kernel must fit inside input, and padding is smaller than input)
        if (t_pad <= jcp.oh * stride_h) {
            // kernel has moved beyond padding (adjust for stride effects)
            if (t_pad % stride_h != 0) {
                assert(!is_dilated);
                int inp_corr = stride_h - t_pad % stride_h;
                add_imm(reg_kernel, reg_kernel, jcp.typesize_out * inp_corr * jcp.kw
                                * jcp.ic_block * jcp.oc_block);
                add_imm(reg_input, reg_input, jcp.typesize_in * inp_corr * iw * inp_mult);
            }
        } else {
            // kernel still overlaps padding (complete reset)
            assert(!is_dilated);
            add_imm(reg_kernel, reg_kernel, -1 * jcp.typesize_out * (t_pad - jcp.oh * stride_h)
                    * jcp.kw * jcp.ic_block * jcp.oc_block);
        }
    }
    assert((jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h) >= 0);
    assert((jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h) < ADDMAX);
    CGA64::cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
    CGA64::b(xa::GE, oh_label_end);
    CGA64::cmp(reg_oj, jcp.oh);
    CGA64::b(xa::GE, oh_label_end);

    /* Compute middle block(s) */
    CGA64::mov(reg_kh, jcp.kh);
    CGA64::L_aarch64(oh_label); {
        compute_oh_step_disp();
        add_imm(reg_input, reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
        add_imm(reg_output, reg_output, jcp.typesize_in * ow * jcp.oc_block);

        add_imm(reg_oj, reg_oj, 1);
        add_imm(reg_ih_count, reg_ih_count, stride_h);

        CGA64::cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
        CGA64::b(xa::GE, oh_label_end);

        CGA64::cmp(reg_oj, jcp.oh);
        CGA64::b(xa::LT, oh_label);
    }
    CGA64::L_aarch64(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        CGA64::cmp(reg_oj, jcp.oh);
        CGA64::b(xa::GE, oh_bpad_label_end);

        if (is_dilated) {
            mov_imm(reg_kh, jcp.kh - 1); //mov(reg_kh, jcp.kh - 1); // assumes unit stride for dilations
            CGA64::mov(reg_tmp, 0);
        } else {
            mov_imm(reg_kh, jcp.ihp - b_pad); //mov(reg_kh, jcp.ihp - b_pad);
            CGA64::sub(reg_kh, reg_kh, reg_ih_count);
        }
        CGA64::L_aarch64(oh_bpad_label);
        {
            compute_oh_step_disp();
            add_imm(reg_input, reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
            add_imm(reg_output, reg_output, jcp.typesize_in * ow * jcp.oc_block);
            if (is_dilated) {
                add_imm(reg_tmp, reg_tmp, 1);
                CGA64::cmp(reg_tmp, dilate_h);
                CGA64::b(xa::LT, oh_dilate_label_end);
                CGA64::mov(reg_tmp, 0);
            }
            add_imm(reg_kh, reg_kh, -1 * stride_h);
            CGA64::cmp(reg_kh, 0);
            CGA64::b(xa::LE,oh_bpad_label_end);
            if (is_dilated)
                CGA64::L_aarch64(oh_dilate_label_end);

            add_imm(reg_oj, reg_oj, 1);
            CGA64::cmp(reg_oj, jcp.oh);
            CGA64::b(xa::LT, oh_bpad_label);
        }
        CGA64::L_aarch64(oh_bpad_label_end);
    }
}

void jit_sve_conv_bwd_weights_kernel_f32::compute_oh_loop_partial() {
    assert(jcp.harness == harness_2d_reduction);
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = jcp.is_1stconv ? 1 : ic_block;
    const int input_bottom_padding_overlap
            = div_up(jcp.ih + jcp.t_pad - (jcp.kh - 1), jcp.stride_h);

    const size_t filter_shift = jcp.typesize_out * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.iw * inp_mult;
    const size_t output_shift = jcp.typesize_out * jcp.ow * oc_block;

    xa::LabelAArch64 loop_begin_label, loop_end_label, common_block_label,
            top_padding_end_label, bottom_padding_end_label,
            bottom_padding_label;

    if (jcp.with_bias) {
        xa::LabelAArch64 skip_zero_bias;
        CGA64::ldr(reg_bias, xa::ptr(param, GET_OFF(bias)));
        CGA64::ldr(reg_tmp, xa::ptr(param, GET_OFF(channel)));
        CGA64::tst(reg_tmp, reg_tmp);
        CGA64::b(xa::EQ, skip_zero_bias);
        CGA64::ldr(reg_tmp, xa::ptr(param, GET_OFF(flags)));
        CGA64::tst(reg_tmp, reg_tmp);
        CGA64::b(xa::NE, skip_zero_bias);
        CGA64::eor(xa::ZRegS(1), reg_p_all_ones.b, xa::ZRegS(1));
        CGA64::str(xa::ZReg(1), xa::ptr(reg_bias)); //vmovups(ptr[reg_bias], Zmm(1));
        CGA64::L_aarch64(skip_zero_bias);
    }

    /* Offset filter position to adjust for top padding */
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(kh_offset)));
    CGA64::add(reg_kernel, reg_kernel, reg_tmp_imm); //add(reg_kernel, ptr[param + GET_OFF(kh_offset)]);

    CGA64::ldr(reg_oj, xa::ptr(param, GET_OFF(os_index_begin)));
    CGA64::ldr(reg_kh, xa::ptr(param, GET_OFF(kh_padding)));

    CGA64::cmp(reg_kh, 0);
    CGA64::b(xa::LE, loop_end_label); // no iterations along kh
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(os_index_end)));
    CGA64::cmp(reg_oj, reg_tmp_imm); //cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
    
    CGA64::b(xa::GE, loop_end_label); // no iterations along height dimension

    CGA64::L_aarch64(loop_begin_label);

    if (jcp.with_bias)
        bias_kernel_2d();
    compute_oh_step_disp();

    /* Compute 'top' edge */
    if (jcp.t_pad > 0) {

        /* Check if within top padding region */
        assert(div_up(jcp.t_pad, jcp.stride_h) >= 0 &&
                div_up(jcp.t_pad, jcp.stride_h) < ADDMAX);
        CGA64::cmp(reg_oj, div_up(jcp.t_pad, jcp.stride_h));
        CGA64::b(xa::GE, top_padding_end_label);

        /* Increment step counter and adjust filter position */
        add_imm(reg_kernel, reg_kernel, -1 * filter_shift * jcp.stride_h);
        add_imm(reg_kh, reg_kh, jcp.stride_h);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kh, jcp.ih);
        mov_imm(reg_tmp_imm, inp_ker_overlap);
        CGA64::cmp(reg_kh, reg_tmp_imm);
        
        CGA64::b(xa::LE, common_block_label);

        /* Correct any excess shifts to kernel and input */
        if (jcp.t_pad <= jcp.oh * jcp.stride_h) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add_imm(reg_kernel, reg_kernel, filter_shift * inp_corr);
                add_imm(reg_input, reg_input, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            add_imm(reg_kernel, reg_kernel, -1 * (jcp.t_pad - jcp.oh * jcp.stride_h) * filter_shift);
        }

        /* Apply correction */
        mov_imm(reg_kh, inp_ker_overlap);
        CGA64::b(common_block_label);

        CGA64::L_aarch64(top_padding_end_label);
    }

    /* Compute 'bottom' edge */
    if (jcp.b_pad > 0) {

        /* Check if within bottom padding region */
        assert((input_bottom_padding_overlap - 1)>=0 &&
                (input_bottom_padding_overlap - 1)<ADDMAX);
        CGA64::cmp(reg_oj, input_bottom_padding_overlap - 1);
        CGA64::b(xa::LT, bottom_padding_end_label);
        CGA64::b(xa::GT, bottom_padding_label);

        /* Execute overlap correction between the filter and the initial
         * bottom padding region. */
        mov_imm(reg_kh, jcp.ih + jcp.t_pad
                        - input_bottom_padding_overlap * jcp.stride_h);
        CGA64::b(bottom_padding_end_label);

        CGA64::L_aarch64(bottom_padding_label);
        add_imm(reg_kh, reg_kh, -1 * jcp.stride_h);
        CGA64::cmp(reg_kh, 0);
        CGA64::b(xa::LE, loop_end_label);

        CGA64::L_aarch64(bottom_padding_end_label);
    }

    /* Compute middle block */
    add_imm(reg_input, reg_input, input_shift * jcp.stride_h);

    /* Execute common block and loop */
    CGA64::L_aarch64(common_block_label);
    add_imm(reg_output, reg_output, output_shift);
    
    add_imm(reg_oj, reg_oj, 1);
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(os_index_end)));
    CGA64::cmp(reg_oj, reg_tmp_imm); //cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
    
    CGA64::b(xa::LT, loop_begin_label);

    CGA64::L_aarch64(loop_end_label);
}

void jit_sve_conv_bwd_weights_kernel_f32::compute_od_loop_partial() {
    assert(jcp.harness == harness_3d_reduction);
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = jcp.is_1stconv ? 1 : ic_block;
    int iw = utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni) ? jcp.tr_iw :
                                                                     jcp.iw;
    int ow = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? jcp.tr_ow : jcp.ow;
    const int input_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const size_t filter_shift
            = jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.ih * iw * inp_mult;
    const size_t output_shift = jcp.typesize_in * jcp.oh * ow * oc_block;

    xa::LabelAArch64 d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    if (jcp.with_bias)
        bias_kernel_3d();

    /* initially offset 'kd' by f_pad */
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(kd_offset)));
    CGA64::add(reg_kernel, reg_kernel, reg_tmp_imm); //add(reg_kernel, ptr[param + GET_OFF(kd_offset)]);

    CGA64::ldr(reg_input_d, xa::ptr(param, GET_OFF(src)));
    CGA64::ldr(reg_output_d, xa::ptr(param, GET_OFF(dst)));
    CGA64::ldr(reg_d_index, xa::ptr(param, GET_OFF(os_index_begin)));
    CGA64::ldr(reg_kd_count, xa::ptr(param, GET_OFF(kd_padding)));

    CGA64::cmp(reg_kd_count, 0);
    CGA64::b(xa::LE, loop_end_label); // no iterations along kd
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(os_index_end)));
    CGA64::cmp(reg_d_index, reg_tmp_imm); //cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    CGA64::b(xa::GE, loop_end_label); // no iterations along depth dimension

    CGA64::L_aarch64(d_loop_label);

    CGA64::mov(reg_input, reg_input_d);
    CGA64::mov(reg_output, reg_output_d);

    push(reg_input_d);
    push(reg_output_d);
    push(reg_d_index);

    compute_oh_loop_common();

    pop(reg_d_index);
    pop(reg_output_d);
    pop(reg_input_d);

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {

        /* Check if within fpad region */
        CGA64::cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        CGA64::b(xa::GE, fpad_end_label);

        /* Fpad steps */
        add_imm(reg_kernel, reg_kernel, -1 * filter_shift * jcp.stride_d);
        add_imm(reg_kd_count, reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kd, jcp.id);
        CGA64::cmp(reg_kd_count, inp_ker_overlap);
        CGA64::b(xa::LE, common_block_label);

        /* Correct any excess shifts to kernel and input */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int inp_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add_imm(reg_kernel, reg_kernel, filter_shift * inp_corr);
                add_imm(reg_input_d, reg_input_d, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            add_imm(reg_kernel, reg_kernel, -1 * (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Apply correction */
        mov_imm(reg_kd_count, inp_ker_overlap);
        CGA64::b(common_block_label);

        CGA64::L_aarch64(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        CGA64::cmp(reg_d_index, input_backpad_overlap - 1);
        CGA64::b(xa::LT, backpad_end_label);
        CGA64::b(xa::GT, backpad_label);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov_imm(reg_kd_count,
                jcp.id + jcp.f_pad - input_backpad_overlap * jcp.stride_d);
        CGA64::b(backpad_end_label);

        CGA64::L_aarch64(backpad_label);
        add_imm(reg_kd_count, reg_kd_count, -1 * jcp.stride_d);
        CGA64::cmp(reg_kd_count, 0);
        CGA64::b(xa::LE, loop_end_label);

        CGA64::L_aarch64(backpad_end_label);
    }

    /* Compute middle block */
    add_imm(reg_input_d, reg_input_d, input_shift * jcp.stride_d);

    /* Execute common block and loop */
    CGA64::L_aarch64(common_block_label);
    add_imm(reg_output_d, reg_output_d, output_shift);
    add_imm(reg_d_index, reg_d_index, 1);
    CGA64::ldr(reg_tmp_imm, xa::ptr(param, GET_OFF(os_index_end)));
    CGA64::cmp(reg_d_index, reg_tmp_imm); //cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    CGA64::b(xa::LT, d_loop_label);

    CGA64::L_aarch64(loop_end_label);
}

void jit_sve_conv_bwd_weights_kernel_f32::compute_loop()
{
    maybe_zero_kernel();

    switch (jcp.harness) {
    case harness_2d_reduction: compute_oh_loop_partial(); break;
    case harness_3d_reduction: compute_od_loop_partial(); break;
    case harness_mb_reduction: compute_oh_loop_common(); break;
    default: assert(!"Invalid harness type");
    }
}

void jit_sve_conv_bwd_weights_kernel_f32::generate()
{
    preamble();
    CGA64::ptrue( reg_p_all_ones.b );

    CGA64::ldr(reg_input, xa::ptr(param, GET_OFF(src)));
    CGA64::ldr(reg_output, xa::ptr(abi_param1_aarch64, GET_OFF(dst)));
    CGA64::ldr(reg_kernel, xa::ptr(abi_param1_aarch64, GET_OFF(filt)));

    compute_loop();

    postamble();
}

status_t jit_sve_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp, const convolution_desc_t &cd,
    cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &diff_weights_pd,
    cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd) {
    if (!mayiuse(sve))
        return status::unimplemented;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper diff_weights_d(&diff_weights_pd);
    const memory_desc_wrapper diff_bias_d(&diff_bias_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();

    jcp.simd_w = cpu_isa_traits<sve>::vlen / sizeof(float);
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims-2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    const int kh_range = 1 + (jcp.kh - 1) * (jcp.dilate_h + 1);
    bool ok = true
        // general condition to simplify dilations
        && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
        && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
        && IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1)
        // special condition to simplify dilations in compute_oh_loop_common
        && IMPLICATION(jcp.dilate_h != 0, kh_range <= jcp.ih);
    if (!ok)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h
            + (jcp.kh - 1) * (jcp.dilate_h + 1) - (jcp.ih + jcp.t_pad - 1));
    jcp.back_pad = nstl::max(0, (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1));

    /* XXX: currently, does not support dilation_d > 0 */
    if (ndims == 5)
        if (jcp.dilate_d > 0)
            return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    /* check for the 1st convolution */
    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = jcp.simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels)
        jcp.oc = rnd_up(jcp.oc, jcp.simd_w);

    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;

    auto src_format = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_format = with_groups
        ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
        : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);
    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format() == any)
            CHECK(diff_bias_pd.set_format(x));
        if (diff_bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.nb_oc = jcp.oc / jcp.oc_block;

    if (diff_dst_d.format() == any)
        CHECK(diff_dst_pd.set_format(src_format));
    if (diff_dst_d.format() != src_format)
        return status::unimplemented;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1) / 2;
    const bool boundaries_ok = true
        && jcp.t_pad <= max_pad
        && jcp.b_pad <= max_pad
        && IMPLICATION(jcp.f_pad > 0, jcp.kd < jcp.id + jcp.f_pad)
        && jcp.f_pad < jcp.kd;
    if (!boundaries_ok)
        return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    if (jcp.is_1stconv) {
        const auto want_src_format = pick(ndims - 3, ncw, nchw, ncdhw);
        if (src_d.format() == any)
            CHECK(src_pd.set_format(want_src_format));

        const bool src_ok = true
            && utils::everyone_is(data_type::f32,
                src_d.data_type(), diff_weights_d.data_type(),
                diff_dst_d.data_type())
            && one_of(jcp.ic, 1, 2, 3)
            && IMPLICATION(jcp.ic == 1, one_of(src_d.format(), want_src_format,
                pick(ndims - 3, nwc, nhwc, ndhwc)))
            && IMPLICATION(jcp.ic != 1, src_d.format() == want_src_format)
            && jcp.ngroups == 1;
        if (!src_ok)
            return status::unimplemented;

        const int tr_ld = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad,
                    jcp.stride_w), 16);
        const int kh_step = nstl::max((28 - jcp.with_bias) / jcp.kw, 1);
        const int kh_step_rem = jcp.kh % kh_step;
        const auto want_4fma_wfmt = with_groups
            ? pick(ndims - 3, gOiw16o, gOihw16o, gOidhw16o)
            : pick(ndims - 3, Oiw16o, Oihw16o, Oidhw16o);

        {
            jcp.ver = ver_fma;
            jcp.ic_block = jcp.ic;

            const auto want_wfmt = with_groups
                ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_wfmt));
            if (diff_weights_d.format() != want_wfmt)
                return status::unimplemented;
        }

        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
    } else {
        if (src_d.format() == any)
            CHECK(src_pd.set_format(src_format));
        if (diff_weights_d.format() == any)
            CHECK(diff_weights_pd.set_format(wei_format));

        const bool ok = true
            && src_d.format() == src_format
            && diff_weights_d.format() == (wei_format);
        if (!ok)
            return status::unimplemented;

        jcp.ic_block = jcp.simd_w;
        if (ok_to_pad_channels)
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
        if ((mayiuse(sve))
                && utils::everyone_is(data_type::f32,
                    src_d.data_type(), diff_weights_d.data_type(),
                    diff_dst_d.data_type())) {
            jcp.ver = ver_fma;
        } else {
            return status::unimplemented;
        }
    }

    if (jcp.ver == ver_fma) {
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
    } else
        return status::unimplemented;

    jcp.harness = ndims == 5 ? harness_3d_reduction : harness_mb_reduction;
    if (jcp.dilate_h == 0 && jcp.ndims == 4 && jcp.oh > min_oh_reduce
            && jcp.ver == ver_fma)
        jcp.harness = harness_2d_reduction; // 2d harness with oh reduction

    bool args_ok = true
        && jcp.ic % jcp.ic_block == 0
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= diff_weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= diff_weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    {   // balancing
        int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);
        jcp.nthr = nthr;
        jcp.nthr_mb = nthr_mb;
        jcp.nthr_g = nthr_g;
        jcp.nthr_oc_b = nthr_oc_b;
        jcp.nthr_ic_b = nthr_ic_b;
    }

    return status::success;
}

void jit_sve_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.nthr_mb > 1) {
        const int wei_size = jcp.ngroups * jcp.oc * jcp.ic
            * jcp.kh * jcp.kw * jcp.kd;
        const int bia_size = jcp.ngroups * jcp.oc;
        const size_t wei_bia_reduction_size = wei_size + bia_size;

        scratchpad.book(key_conv_wei_bia_reduction,
                jcp.typesize_out * wei_bia_reduction_size * (jcp.nthr_mb - 1));
        scratchpad.book(key_conv_wei_bia_reduction_bctx,
                sizeof(simple_barrier::ctx_t));
    }

    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.typesize_out * jcp.oc);
}

void jit_sve_conv_bwd_weights_kernel_f32::balance(
        const jit_conv_conf_t &j, int &nthr_, int &nthr_mb_, int &nthr_g_,
        int &nthr_oc_b_, int &nthr_ic_b_)
{
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    const int max_threads = mkldnn_get_max_threads();

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    int ih_reduce = j.harness == harness_2d_reduction ? j.ih : 1;
    int oh_reduce = j.harness == harness_2d_reduction ? j.oh : 1;
    int ih_no_reduce = j.harness == harness_2d_reduction ? 1 : j.ih;
    int oh_no_reduce = j.harness == harness_2d_reduction ? 1 : j.oh;
    int nthr_oh_reduce = nstl::max(1, oh_reduce / min_oh_reduce);

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */

        const int src_coef = 1;
        const int dst_coef = 1;
        const int wei_coef = 8;

        return 0
            + src_coef
            * div_up(j.mb * ih_reduce, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_ic, nthr_ic_b) * j.ic_block * ih_no_reduce * j.iw * j.id
            / j.stride_d / j.stride_h / j.stride_w /* (n1) */
            + dst_coef
            * div_up(j.mb * oh_reduce, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * j.oc_block * oh_no_reduce * j.ow * j.od
            + wei_coef /* (n2) */
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b)
            * j.kh * j.kw * j.kd * j.ic_block * j.oc_block;
    };

    int best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb * j.od * nthr_oh_reduce);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);

            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }

        if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
    }

    {
        auto calc_comp_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
            return 1
                * div_up(j.mb * oh_reduce, nthr_mb)
                * div_up(j.ngroups, nthr_g_)
                * div_up(j.nb_oc, nthr_oc_b)
                * div_up(j.nb_ic, nthr_ic_b);
        };

        /* step 2: search for a thread distribution with lower compute cost.
         * the constrains:
         *  - memory cost cannot exceed 110% of the best found in the step 1
         *  - unless compute cost is 133% lower than the current best case
         * note: both constants were found empirically */
        int best_comp_cost = calc_comp_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);
        for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
            const int nthr_par = nthr / nthr_mb;
            const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
            for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
                int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);
                int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
                int comp_cost = calc_comp_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

                const bool opt1 = comp_cost <= best_comp_cost
                    && mem_cost < 1.1 * best_mem_cost;
                const bool opt2 = 4 * comp_cost <= 3 * best_comp_cost;

                if (opt1 || opt2) {
                    best_comp_cost = comp_cost;
                    nthr_mb_ = nthr_mb;
                    nthr_oc_b_ = nthr_oc_b;
                    nthr_ic_b_ = nthr_ic_b;
                }
            }

            if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
        }
    }

    if (nthr_mb_ > max_threads / 2 && nthr_mb_ < max_threads)
        nthr_mb_ = nstl::min(j.mb * j.od * nthr_oh_reduce, max_threads);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= max_threads);
    assert(IMPLICATION(!mkldnn_thr_syncable(), nthr_mb_ == 1));
}

template struct  _jit_sve_conv_fwd_kernel<Zmm>;
template struct  _jit_sve_conv_fwd_kernel<Xmm>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
