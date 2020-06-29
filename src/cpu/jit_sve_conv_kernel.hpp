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
* Copyright 2016-2018 Intel Corporation
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

#ifndef JIT_SVE_CONV_KERNEL_F32_HPP
#define JIT_SVE_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_memory.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"


#define PRFWMAX    31
#define LDRMAX    255
#define LDRWMAX   252
#define ADDMAX   4095
#define PRFMMAX 32760
#define MOVMAX  65535

namespace mkldnn {
namespace impl {
namespace cpu {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;
/* Get vector offsets, ofs / VL(VL: 512bits = 64Bytes) */
#define VL_OFS(ofs) ((ofs)>>6)

template<typename Vmm>
struct _jit_sve_conv_fwd_kernel : public jit_generator {

    _jit_sve_conv_fwd_kernel(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);

        generate();
        jit_ker_ = (void (*)(jit_conv_call_s *))getCode32();
    }

    ~_jit_sve_conv_fwd_kernel() {
        delete eltwise_injector_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_sve_conv_fwd_kernel)

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    using reg64_t = const xa::XReg;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    const xa::PReg reg_p_all_ones  = p2;

    reg64_t param               = abi_param1_aarch64;
    reg64_t reg_inp             = x1;  
    reg64_t reg_ker             = x2;  
    reg64_t reg_out             = x3;  
    reg64_t reg_inp_prf         = x20;   
    reg64_t reg_ker_prf         = x5;  
    reg64_t reg_owb             = x5;  
    reg64_t reg_out_prf         = x6;  

    reg64_t aux_reg_inp         = x7;  
    reg64_t aux_reg_ker         = x8;  
    reg64_t aux_reg_inp_prf     = x9;  
    reg64_t aux_reg_ker_prf     = x10; 
    reg64_t reg_channel         = x9;
    reg64_t reg_bias            = x10;

    reg64_t aux_reg_ker_d       = x2;
    reg64_t aux_reg_inp_d       = x11;
    reg64_t aux_reg_inp_d_prf   = x6;
    reg64_t aux_reg_ker_d_prf   = x12;
    reg64_t reg_ki              = x3;

    reg64_t reg_kj              = x13; 
    reg64_t reg_relu_ns         = x13; 
    reg64_t reg_oi              = x11; 
    reg64_t reg_kh              = x12; 
    reg64_t reg_ic_loop         = x10; 
    reg64_t reg_inp_loop        = x9;  
    reg64_t reg_init_flag       = x6;  

    reg64_t aux_reg_ic          = x5;  
    reg64_t reg_binp            = x13; 
    reg64_t reg_bout            = x20;   
    reg64_t aux1_reg_inp        = x11; 
    reg64_t aux_reg_out         = x12; 
    reg64_t reg_long_offt       = x20;   
    reg64_t reg_out_long_offt   = x7;  

    /* Temporary registers for ARM insts */
    reg64_t reg_tmp_addr        = x14;
    reg64_t reg_prev_bcast_addr = x15;
    reg64_t reg_prev_wei_addr   = x16;
    reg64_t reg_tmp_imm         = x17;

    reg64_t reg_out_org         = x18;
    reg64_t reg_oi_org          = x19;

    void prefetch(const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_alinged = ((ofs&0xFF)==0) ? true : false;
        if (cacheline_alinged == true) {
            xa::Prfop op = xa::PLDL1KEEP;
            switch (level) {
            case 1: op = (for_load == true) ? xa::PLDL1KEEP : xa::PSTL1KEEP; break;
            case 2: op = (for_load == true) ? xa::PLDL2KEEP : xa::PSTL2KEEP; break;
            case 3: op = (for_load == true) ? xa::PLDL3KEEP : xa::PSTL3KEEP; break;
            default: assert(!"invalid prfop"); break;
            }

            if((ofs <= PRFMMAX) && (ofs >= 0)) {
                CGA64::prfm(op, xa::ptr(in, static_cast<int32_t>(ofs)));
            }else{
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfm(op, xa::ptr(reg_tmp_addr));
            }
        } else {
            xa::PrfopSve op_sve = xa::PLDL1KEEP_SVE;
            switch (level) {
            case 1: op_sve = (for_load == true) ? xa::PLDL1KEEP_SVE : xa::PSTL1KEEP_SVE; break;
            case 2: op_sve = (for_load == true) ? xa::PLDL2KEEP_SVE : xa::PSTL2KEEP_SVE; break;
            case 3: op_sve = (for_load == true) ? xa::PLDL3KEEP_SVE : xa::PSTL3KEEP_SVE; break;
            default: assert(!"invalid level"); break;
            }

            if((VL_OFS(ofs) <= PRFWMAX) &&
               (VL_OFS(ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(in, static_cast<int32_t>(VL_OFS(ofs))));
            }else{
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(reg_tmp_addr));
            }
        }
    }

    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma_core(int ur_w, int pad_l, int pad_r);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate();
    inline size_t get_output_offset(int oi, int n_oc_block) {
        return (size_t)jcp.typesize_out * ((size_t)n_oc_block * jcp.oh
            * jcp.ow * jcp.od + oi) * jcp.oc_block;
    }

    inline size_t get_input_offset(int ki, int ic, int oi, int pad_l) {
        size_t iw_str = !jcp.is_1stconv ? jcp.ic_block : 1;
        size_t ic_str = !jcp.is_1stconv ? 1 : (size_t)jcp.iw * jcp.ih * jcp.id;
        return (size_t)jcp.typesize_in
                * ((size_t)(ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l)
                                  * iw_str
                          + ic * ic_str);
    }

    inline int get_kernel_offset(int ki,int ic,int n_oc_block,int ker_number) {
        return jcp.typesize_in * jcp.oc_block
            * (n_oc_block * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd
                    + (ic + ker_number) + ki * jcp.ic_block);
    }

    inline int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    inline int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w - nstl::max(0, utils::div_up(pad_r
                                                   - (jcp.kw - 1 - ki)
                                                           * (jcp.dilate_w + 1),
                                           jcp.stride_w));
    }
};

struct jit_sve_conv_fwd_kernel {

    jit_sve_conv_fwd_kernel(jit_conv_conf_t ajcp,
        const primitive_attr_t &attr) :
        jit_ker(nullptr),
        zmm_kernel_(nullptr),
        xmm_kernel_(nullptr) {
        int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.oc_block;
        switch (ch_block) {
        case 16:
            zmm_kernel_ =
                new _jit_sve_conv_fwd_kernel<Xbyak::Zmm>(
                    ajcp, attr);
            jit_ker = zmm_kernel_->jit_ker_;
            return;
        case 4:
            xmm_kernel_ =
                new _jit_sve_conv_fwd_kernel<Xbyak::Xmm>(
                    ajcp, attr);
            jit_ker = xmm_kernel_->jit_ker_;
            return;
        default:
            assert(!"invalid channel blocking");
        }
    }

    ~jit_sve_conv_fwd_kernel() {
        delete xmm_kernel_;
        delete zmm_kernel_;
    }

    enum {
        typesize = sizeof(float)
    };

    static bool post_ops_ok(jit_conv_conf_t &jcp,
        const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        cpu_memory_t::pd_t &src_pd,
        cpu_memory_t::pd_t &weights_pd,
        cpu_memory_t::pd_t &dst_pd,
        cpu_memory_t::pd_t &bias_pd,
        const primitive_attr_t &attr,
        int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_conv_conf_t &jcp);

    void(*jit_ker)(jit_conv_call_s *);
    _jit_sve_conv_fwd_kernel<Xbyak::Zmm> *zmm_kernel_;
    _jit_sve_conv_fwd_kernel<Xbyak::Xmm> *xmm_kernel_;
};

struct jit_sve_conv_bwd_data_kernel_f32: public jit_generator {

    jit_sve_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode32();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_conv_bwd_data_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const xa::XReg;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 26,
    };

    reg64_t param               = abi_param1_aarch64;
    reg64_t reg_dst             = x1;
    reg64_t reg_ker             = x2;
    reg64_t reg_src             = x3;

    reg64_t reg_dst_prf         = x23;
    reg64_t reg_ker_prf         = x5;
    reg64_t reg_src_prf         = x6;

    reg64_t aux_reg_dst         = x7;
    reg64_t aux_reg_ker         = x8;

    reg64_t aux_reg_dst_prf     = x9;
    reg64_t aux_reg_ker_prf     = x10;

    reg64_t aux_reg_dst_d_prf   = x6;
    reg64_t aux_reg_dst_d       = x11;
    reg64_t aux_reg_ker_d_prf   = x12;
    reg64_t aux_reg_ker_d       = x2;
    reg64_t reg_ki              = x3;

    reg64_t reg_kj              = x13;
    reg64_t reg_oi              = x11;
    reg64_t reg_kh              = x12;

    reg64_t reg_channel         = x9;

    reg64_t reg_tmp             = x14;
    reg64_t reg_long_offt       = x7;

    /* Temporary registers for ARM insts */
    reg64_t reg_prev_bcast_addr = x15; 
    reg64_t reg_tmp_imm         = x16; 
    reg64_t reg_tmp_addr        = x18; 

    reg64_t reg_src_prf_org     = x19;
    reg64_t reg_src_org         = x20;
    reg64_t reg_oi_org          = x21;

    const xa::PReg reg_p_all_ones  = p2;

    void prefetch(const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_alinged = ((ofs&0xFF)==0) ? true : false;
        if (cacheline_alinged == true) {
            xa::Prfop op;
            switch (level) {
            case 1: op = (for_load == true) ? xa::PLDL1KEEP : xa::PSTL1KEEP; break;
            case 2: op = (for_load == true) ? xa::PLDL2KEEP : xa::PSTL2KEEP; break;
            case 3: op = (for_load == true) ? xa::PLDL3KEEP : xa::PSTL3KEEP; break;
            default: assert(!"invalid prfop"); break;
            }

            if((ofs <= PRFMMAX) && (ofs >= 0)) {
              CGA64::prfm(op, xa::ptr(in, static_cast<int32_t>(ofs)));
            }else{
              CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
              CGA64::prfm(op, xa::ptr(reg_tmp_addr));
            }
        } else {
            xa::PrfopSve op_sve;
            switch (level) {
            case 1: op_sve = (for_load == true) ? xa::PLDL1KEEP_SVE : xa::PSTL1KEEP_SVE; break;
            case 2: op_sve = (for_load == true) ? xa::PLDL2KEEP_SVE : xa::PSTL2KEEP_SVE; break;
            case 3: op_sve = (for_load == true) ? xa::PLDL3KEEP_SVE : xa::PSTL3KEEP_SVE; break;
            default: assert(!"invalid prfop"); break;
            }

            if((VL_OFS(ofs) <= PRFWMAX) &&
               (VL_OFS(ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(in, static_cast<int32_t>(VL_OFS(ofs))));
            }else{
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(reg_tmp_addr));
            }
        }
    }

    xa::ZReg reg_wei = xa::ZReg(31);

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma_core(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate();

    inline int get_iw_start(int ki, int l_overflow)
    {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow)
    {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }
};

struct jit_sve_conv_bwd_weights_kernel_f32 : public jit_generator {

    jit_sve_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp)
        : jit_generator(nullptr, 1024*1024), jcp(ajcp)
    {
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode32();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_conv_bwd_weights_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &diff_weights_pd,
            cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const xa::XReg;
    enum {typesize = sizeof(float)};
    static const int max_ur_w;
    static const int min_oh_reduce;

    reg64_t param          = abi_param1_aarch64;
    reg64_t reg_input      = x1;
    reg64_t reg_kernel     = x2;
    reg64_t reg_output     = x3;
    reg64_t b_ic           = x20;
    reg64_t kj             = x5;
    reg64_t reg_kh         = x6;
    reg64_t reg_ur_w_trips = x7;
    reg64_t reg_oj         = x8;
    reg64_t reg_ih_count   = x9;
    reg64_t reg_tmp        = x10;
    reg64_t reg_long_offt  = x10;

    reg64_t ki             = x11;
    reg64_t reg_kd_count   = x12;
    reg64_t reg_oi         = x12;
    reg64_t reg_d_index    = x13;
    reg64_t reg_input_d    = x8;
    reg64_t reg_output_d   = x9;
    reg64_t aux_reg_input  = x12;
    reg64_t aux_reg_kernel = x13;
    reg64_t reg_bias       = x9;

    reg64_t reg_add_tmp    = x14;
    reg64_t reg_tmp_imm    = x15;

    reg64_t reg_kd_count_org = x16;
    reg64_t reg_input_d_org  = x17;
    reg64_t reg_output_d_org = x18;
    reg64_t reg_d_index_org  = x19;

    xa::ZRegS zreg_idata   = xa::ZRegS(31);

    const xa::PReg reg_p_all_ones = p2;

    void prefetch(const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_alinged = ((ofs&0xFF)==0) ? true : false;
        if (cacheline_alinged == true) {
            xa::Prfop op;
            switch (level) {
            case 1: op = (for_load == true) ? xa::PLDL1KEEP : xa::PSTL1KEEP; break;
            case 2: op = (for_load == true) ? xa::PLDL2KEEP : xa::PSTL2KEEP; break;
            case 3: op = (for_load == true) ? xa::PLDL3KEEP : xa::PSTL3KEEP; break;
            default: assert(!"invalid prfop"); break;
            }

            if((ofs <= PRFMMAX) && (ofs >= 0)) {
                CGA64::prfm(op, xa::ptr(in, static_cast<int32_t>(ofs)));
            }else{
                CGA64::add_imm(reg_add_tmp, in, ofs, reg_tmp_imm);
                CGA64::prfm(op, xa::ptr(reg_add_tmp));
            }
        } else {
            xa::PrfopSve op_sve;
            switch (level) {
            case 1: op_sve = (for_load == true) ? xa::PLDL1KEEP_SVE : xa::PSTL1KEEP_SVE; break;
            case 2: op_sve = (for_load == true) ? xa::PLDL2KEEP_SVE : xa::PSTL2KEEP_SVE; break;
            case 3: op_sve = (for_load == true) ? xa::PLDL3KEEP_SVE : xa::PSTL3KEEP_SVE; break;
            default: assert(!"invalid prfop"); break;
            }

            if((VL_OFS(ofs) <= PRFWMAX) &&
               (VL_OFS(ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(in, static_cast<int32_t>(VL_OFS(ofs))));
            }else{
                CGA64::add_imm(reg_add_tmp, in, ofs, reg_tmp_imm);
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(reg_add_tmp));
            }
        }
    }

    inline void bias_kernel_2d();
    inline void bias_kernel_3d();
    inline void maybe_zero_kernel();
    inline void compute_oh_step_unroll_ow_icblock(int ic_block_step,
            int max_ur_w);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_ic_block_step(int ur_w,
            int pad_l, int pad_r, int ic_block_step,
            int input_offset, int kernel_offset, int output_offset,
            bool input_wraparound = false);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_disp();
    inline void compute_oh_loop_common();
    inline void compute_oh_loop_partial();
    inline void compute_od_loop_partial();

    inline void compute_loop();

    void generate();

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b);
};

}
}
}

#endif
