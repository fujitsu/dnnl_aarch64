/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef JIT_SVE_1x1_CONV_KERNEL_HPP
#define JIT_SVE_1x1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

using namespace mkldnn::impl::types;

#define PRFWMAX   32
#define LDRMAX   256
#define LDRWMAX  253
#define ADDMAX  4096
#define MOVMAX 65536

namespace mkldnn {
namespace impl {
namespace cpu {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

struct jit_sve_1x1_conv_kernel : public jit_generator {
    jit_sve_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp,
            const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *)) this->getCode32();

    }

    ~jit_sve_1x1_conv_kernel() {
        delete eltwise_injector_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_1x1_conv_kernel)

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
                                const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const primitive_attr_t &attr,
            int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

  private:
    using reg64_t = const xa::XReg;
    const xa::PReg reg_p_all_ones  = p1;

    /* Flag */
    reg64_t reg_reduce_pos_flag     = x8; 
    reg64_t aux1_reg_bcast_data     = x9; 

    reg64_t aux_reg_output_data     = x10;
    reg64_t reduce_loop_iter        = x11;
    reg64_t bcast_loop_iter         = x12;

    reg64_t reg_relu_ns             = x13; // For forward
    reg64_t reg_output_stride       = x13; // For backward

    reg64_t aux_reg_bcast_data      = x14;
    reg64_t aux_reg_load_data       = x15;

    /* Pointer */
    reg64_t reg_bcast_data          = x16; // Weight
    reg64_t reg_load_data           = x17; // Input
    reg64_t reg_output_data         = x18; // Output
    reg64_t reg_bias_data           = x19; // bias

    reg64_t reg_load_data_tmp       = x20; // Weight
    reg64_t reg_prev_bcast_addr     = x21; // Input
    reg64_t reg_bias_data_tmp       = x22; // Bias
    reg64_t reg_tmp                 = x23; // tmp for add_imm
    reg64_t reg_tmp_ofs             = x23; // tmp_ofs (for load, bcast, output, bias_ofs, generate())
    reg64_t reg_prev_out_addr       = x24; // this reg keeps addr accessed by previous ldr or str inst

    /* Workload */
    reg64_t reg_load_loop_work      = x27;
    reg64_t reg_reduce_loop_work    = x29;
    reg64_t reg_bcast_loop_work     = x30;

    reg64_t imm_addr64              = aux_reg_load_data;

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


#if 1
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;
#else
    void *eltwise_injector_;
#endif

    int bcast_loop_work_offt = 0;
    int stack_space_needed = 16;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    static void balance(jit_1x1_conv_conf_t &jcp, int nthreads);
};

}
}
}

#endif
