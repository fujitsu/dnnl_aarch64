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

#ifndef JIT_SVE_X8S8S32X_1X1_CONV_KERNEL_HPP
#define JIT_SVE_X8S8S32X_1X1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

#define ADDMAX  4095
#define MOVMAX 65535

namespace mkldnn {
namespace impl {
namespace cpu {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

struct jit_sve_x8s8s32x_1x1_conv_kernel: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_x8s8s32x_1x1_conv_fwd_ker_t)
    jit_sve_x8s8s32x_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp,
            const primitive_attr_t &attr) : jcp(ajcp), attr_(attr),
            eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *)) this->getCode();
    }

    ~jit_sve_x8s8s32x_1x1_conv_kernel() {
        delete eltwise_injector_;
    }

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
                                const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &bias_d,
            const primitive_attr_t &attr,
            int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    bool maybe_eltwise(int position);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

  private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;

    reg64_t reg_bcast_data = r8;
    reg64_t reg_ptr_scales = r8;
    reg64_t reg_output_data = r9;
    reg64_t reg_load_data = r10;
    reg64_t reg_ptr_sum_scale = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t reg_bias_data = r12;
    reg64_t reg_comp_data = r12;
    reg64_t reg_scratch = r13;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux_reg_load_data = r15;
    reg64_t imm_addr64 = r15;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t reg_bcast_loop_work = rbx;
    reg64_t bcast_loop_iter = rdx; // Note: Fix me
    reg64_t reg_load_loop_work = rsi;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reduce_loop_iter = abi_param1;

    /* Temporay registers */
    xa::XReg reg_tmp_imm = x18; // tmp for add_imm
    xa::XReg reg_tmp_adr = x19; // tmp for address value

    reg64_t reg_last_load = r8;
    mask_t ktail_mask = k6;

    mask_t vmask = k7;

    Xbyak::Zmm zmm_tmp = Xbyak::Zmm(28);
    Xbyak::Zmm zmm_one = Xbyak::Zmm(29);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_bcast = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_bcast2 = Xbyak::Zmm(30);

    int bcast_loop_work_off = 0;
    int reg_bias_data_off = 8;
    int reg_bcast_data_off = 16;
    int reg_load_data_off = 24;
    int reg_ptr_sum_scale_off = 32;
    int reg_comp_data_off = 40;
    int stack_space_needed = 48;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    void cvt2ps(data_type_t type_in, zmm_t zmm_in, const Xbyak::Operand &op,
        bool mask_flag);
    Xbyak::Address SVE_compress_addr(Xbyak::Reg64 base, int raw_offt)
    {
        using Xbyak::Zmm;
        using Xbyak::Reg64;
        using Xbyak::Address;
        using Xbyak::RegExp;

        assert(raw_offt <= INT_MAX);
        auto offt = static_cast<int>(raw_offt);

        int scale = 0;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = base + offt;
        if (scale)
            re = re + (2 * EVEX_max_8b_offt) * scale;

        return zword [re];
    }

    void add_imm(xa::XReg out, xa::XReg in, long long int value){
        long long int val = (value >= 0) ? value : -1 * value;
        if( val <= ADDMAX ){
            if( value >= 0 )  CGA64::add(out, in, val);
            else              CGA64::sub(out, in, val);
        }else{
            CGA64::mov(reg_tmp_imm, val&0xffff);
            if(val > MOVMAX) CGA64::movk(reg_tmp_imm, (val>>16)&0xffff, 16);
            if(val > 0xffffffff) CGA64::movk(reg_tmp_imm, (val>>32)&0xffff, 32);
            if(val > 0xffffffffffff) CGA64::movk(reg_tmp_imm, (val>>48)&0xffff, 48);

            if( value >= 0 )  CGA64::add(out, in, reg_tmp_imm);
            else              CGA64::sub(out, in, reg_tmp_imm);
        }
    }
};

}
}
}

#endif
