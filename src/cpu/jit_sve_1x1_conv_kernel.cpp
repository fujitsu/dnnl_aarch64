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

#include <assert.h>
#include <float.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"
#include "cpu_barrier.hpp"

#include "jit_sve_1x1_conv_utils.hpp"
#include "jit_sve_1x1_conv_kernel.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_1x1_conv_call_s, field))

using namespace mkldnn::impl::types;

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

void jit_sve_1x1_conv_kernel::bcast_loop(int load_loop_blk)
{
    CGA64::mov(aux1_reg_bcast_data, reg_bcast_data);
    CGA64::mov(aux_reg_bcast_data, reg_bcast_data);

    CGA64::mov(aux_reg_output_data, reg_output_data);
    // CGA64::mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_offt));
    CGA64::mov(bcast_loop_iter, reg_bcast_loop_work);


    xa::LabelAArch64 bcast_loop;
    xa::LabelAArch64 bcast_loop_tail;

    CGA64::cmp(bcast_loop_iter, jcp.ur);
    CGA64::b(xa::LT, bcast_loop_tail);

    CGA64::L_aarch64(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
  
                add_imm(aux_reg_output_data, aux_reg_output_data, jcp.bcast_loop_output_substep);
            }
            else {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                    - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);

                add_imm(aux_reg_output_data, aux_reg_output_data, jcp.bcast_loop_output_step
                    - (num_substeps - 1) * jcp.bcast_loop_output_substep);
            }
        }
        CGA64::sub(bcast_loop_iter, bcast_loop_iter, jcp.bcast_block);
        CGA64::cmp(bcast_loop_iter, jcp.bcast_block);
        CGA64::b(xa::GE, bcast_loop);
    }

    CGA64::L_aarch64(bcast_loop_tail);
    if (jcp.ur_tail) {
        xa::LabelAArch64 bcast_loop_tail_out;
        CGA64::cmp(bcast_loop_iter, 0);
        CGA64::b(xa::EQ, bcast_loop_tail_out);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        CGA64::L_aarch64(bcast_loop_tail_out);
    }
}

void jit_sve_1x1_conv_kernel::reduce_loop(int load_loop_blk,
         int ur, int substep, bool wraparound)
{
    auto vreg_bcast_s = [=]() {
        return xa::ZRegS(30);
    };

    auto vreg_sum = [=]() {
        return xa::ZReg(31);
    };
    auto vreg_sum_s = [=]() {
        return xa::ZRegS(31);
    };

    auto vreg_load = [=](int i_load, int i_fma) {
        return xa::ZReg(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };
    auto vreg_load_s = [=](int i_load, int i_fma) {
        return xa::ZRegS(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return xa::ZReg(i_ur * load_loop_blk + i_load);
    };
    auto vreg_accum_s = [=](int i_load, int i_ur) {
        return xa::ZRegS(i_ur * load_loop_blk + i_load);
    };

    auto bias_load = [=](int i_load, int i_ur){
      if (one_of(jcp.prop_kind, forward_training, forward_inference,
                 backward_data)){
        int ofs = jcp.typesize_out * jcp.oc_block * i_load;

        if ( (ofs>>6) < LDRMAX ){
          CGA64::ldr(vreg_accum(i_load, i_ur), xa::ptr(reg_bias_data, static_cast<int32_t>(ofs>>6)));
        }else if( ofs < ADDMAX){
          CGA64::add(reg_bias_data_tmp, reg_bias_data, ofs);
          CGA64::ldr(vreg_accum(i_load, i_ur), xa::ptr(reg_bias_data_tmp));
        }else if( ofs < MOVMAX){
          CGA64::mov( reg_tmp_ofs, ofs );
          CGA64::add(reg_bias_data_tmp, reg_bias_data, reg_tmp_ofs );
          CGA64::ldr(vreg_accum(i_load, i_ur), xa::ptr(reg_bias_data_tmp));
        }else{
          CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
          CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
          CGA64::add(reg_bias_data_tmp, reg_bias_data, reg_tmp_ofs );
          CGA64::ldr(vreg_accum(i_load, i_ur), xa::ptr(reg_bias_data_tmp));
        }

        return reg_tmp_ofs;
      }else
        assert(NULL); // TODO

    };

    auto init = [=]() {
        xa::LabelAArch64 init_done;
        xa::LabelAArch64 init_zero;

        /* Read bias */
        if (jcp.with_bias
            && one_of(jcp.prop_kind, forward_training, forward_inference)) {

            tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            CGA64::b(xa::EQ, init_zero);

            // TODO: We need impl offset calc part in the following loop
            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur){
                    bias_load(i_load, i_ur);
                }
            CGA64::b(init_done);
        }

        CGA64::L_aarch64(init_zero);
        /* Zero clear */
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                CGA64::fmov(vreg_accum_s( i_load, i_ur ));
            }
        CGA64::L_aarch64(init_done);
    };


    auto bcast_load = [=] (int i_reduce, int i_ur, int prev_ofs){

      int ofs;

      if (one_of(jcp.prop_kind, forward_training, forward_inference,
                 backward_data)) {

        ofs = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
                    : i_ur * jcp.reduce_loop_unroll + i_reduce;

      }else{
        if (jcp.transpose_src) {
          const int reduce_group = i_reduce / 4;
          const int reduce_shift = i_reduce % 4;
          ofs = 4 * (reduce_group * jcp.ic_block + i_ur) + reduce_shift;
        }
        else
          ofs = i_reduce * jcp.ic_block + i_ur;
      }

      ofs = jcp.typesize_in * ofs;

      if( ((ofs&0x3) == 0) && (ofs < LDRWMAX)){
        ld1rw(vreg_bcast_s(), reg_p_all_ones.b, xa::ptr(aux_reg_bcast_data, static_cast<int32_t>(ofs)));
      }else{
        if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&((ofs - prev_ofs) < LDRMAX) && (((ofs-prev_ofs)&0x3) == 0)){
          ld1rw(vreg_bcast_s(), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr, static_cast<int32_t>((ofs-prev_ofs))));
        }else{
          if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
            ofs = ofs - prev_ofs;
            if( ofs < ADDMAX){
              CGA64::add(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs);
            }else if( ofs < MOVMAX ){
              CGA64::mov(reg_tmp_ofs, ofs);
              CGA64::add(reg_prev_bcast_addr, reg_prev_bcast_addr, reg_tmp_ofs);
            }else{
              CGA64::mov( reg_tmp_ofs, ofs&0xffff);
              CGA64::movk( reg_tmp_ofs, ofs>>16, 16);
              CGA64::add(reg_prev_bcast_addr, reg_prev_bcast_addr, reg_tmp_ofs);
            }

          }else{
            if( ofs < ADDMAX){
              CGA64::add(reg_prev_bcast_addr, aux_reg_bcast_data, ofs);
            }else if( ofs < MOVMAX ){
              CGA64::mov(reg_tmp_ofs, ofs);
              CGA64::add(reg_prev_bcast_addr, aux_reg_bcast_data, reg_tmp_ofs);
            }else{
              CGA64::mov( reg_tmp_ofs, ofs&0xffff);
              CGA64::movk( reg_tmp_ofs, ofs>>16, 16);
              CGA64::add(reg_prev_bcast_addr, aux_reg_bcast_data, reg_tmp_ofs);
            }
          }
          prev_ofs = (i_reduce == jcp.reduce_loop_unroll)
                      ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
                      : i_ur * jcp.reduce_loop_unroll + i_reduce;
          prev_ofs = jcp.typesize_in * prev_ofs;

          CGA64::ld1rw(vreg_bcast_s(), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr));
        }
      }

      return prev_ofs;

    };

    auto load_load = [=]( int i_reduce, int i_load, int i_fma ){
      int ofs;
      int u0 = i_reduce % jcp.reduce_loop_unroll;
      int u1 = i_reduce / jcp.reduce_loop_unroll;
      ofs = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
      ofs = u1 * jcp.reduce_loop_load_step + jcp.typesize_in * ofs;

      if(((ofs >> 6 ) < LDRMAX) && ((ofs&0x3f) == 0) ){
        ofs = ofs >> 6;
        CGA64::ldr(vreg_load(i_load, i_fma), xa::ptr(aux_reg_load_data, static_cast<int32_t>(ofs)));
      }else if(ofs < ADDMAX){
        CGA64::add(reg_load_data_tmp, aux_reg_load_data, ofs);
        CGA64::ldr(vreg_load(i_load, i_fma), xa::ptr(reg_load_data_tmp));
      }else if(ofs < MOVMAX){
        CGA64::mov(reg_load_data_tmp, ofs);
        CGA64::add(reg_load_data_tmp, aux_reg_load_data, reg_load_data_tmp);
        CGA64::ldr(vreg_load(i_load, i_fma), xa::ptr(reg_load_data_tmp));
      }else{
        CGA64::mov(reg_load_data_tmp, ofs&0xffff);
        CGA64::movk(reg_load_data_tmp, ofs>>16, 16);
        CGA64::add(reg_load_data_tmp, aux_reg_load_data, reg_load_data_tmp);
        CGA64::ldr(vreg_load(i_load, i_fma), xa::ptr(reg_load_data_tmp));
      }
    };

    auto out_load = [=](int i_load, int i_ur, int prev_ofs){
      int ofs;
      if (one_of(jcp.prop_kind, forward_training, forward_inference,
                 backward_data)){
        ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block * jcp.typesize_out;
        if((ofs>>6) < LDRMAX){
          CGA64::ldr(vreg_sum(), xa::ptr(aux_reg_output_data, static_cast<int32_t>(ofs>>6)));
        }else{
          if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&(((ofs - prev_ofs)>>6) < LDRMAX)){
            CGA64::ldr(vreg_sum(), xa::ptr(reg_prev_out_addr, static_cast<int32_t>((ofs - prev_ofs)>>6)));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
              ofs = ofs - prev_ofs;
              if( ofs < ADDMAX){
                CGA64::add(reg_prev_out_addr, reg_prev_out_addr, ofs);
              }else if( ofs < MOVMAX){
                CGA64::mov( reg_tmp_ofs, ofs);
                CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
              }else{
                CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
              }
            }else{
              if( ofs < ADDMAX){
                CGA64::add(reg_prev_out_addr, aux_reg_output_data, ofs);
              }else if( ofs < MOVMAX){
                CGA64::mov( reg_tmp_ofs, ofs);
                CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
              }else{
                CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
              }
            }
  
            CGA64::ldr(vreg_sum(), xa::ptr(reg_prev_out_addr));
  
            prev_ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block * jcp.typesize_out;
          }
        }
 
      }else{
        //(i_load ? reg_output_stride * i_load : 0) 
        ofs = jcp.typesize_out * jcp.load_block * i_ur;
        if( i_load ){
          CGA64::mov(reg_tmp_out_ofs, i_load);
          if((ofs>>6) < LDRMAX){
            CGA64::madd(reg_tmp_out_ofs, reg_tmp_out_ofs, reg_output_stride, aux_reg_output_data);
            CGA64::ldr(vreg_sum(), xa::ptr(reg_tmp_out_ofs, static_cast<int32_t>(ofs>>6)));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&(((ofs - prev_ofs)>>6) < LDRMAX)){
              CGA64::madd(reg_tmp_out_ofs, reg_tmp_out_ofs, reg_output_stride, reg_prev_out_addr);
              CGA64::ldr(vreg_sum(), xa::ptr(reg_tmp_out_ofs, static_cast<int32_t>((ofs - prev_ofs)>>6)));
            }else{
              if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
                ofs = ofs - prev_ofs;
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, reg_prev_out_addr, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }
              }else{
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, aux_reg_output_data, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }
              }
              CGA64::madd(reg_tmp_out_ofs, reg_tmp_out_ofs, reg_output_stride, reg_prev_out_addr);
              CGA64::ldr(vreg_sum(), xa::ptr(reg_tmp_out_ofs));
    
              prev_ofs = jcp.typesize_out * jcp.load_block * i_ur;
            }
          }
        }else{
          if((ofs>>6) < LDRMAX){
            CGA64::ldr(vreg_sum(), xa::ptr(aux_reg_output_data, static_cast<int32_t>(ofs>>6)));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&(((ofs - prev_ofs)>>6) < LDRMAX)){
              CGA64::ldr(vreg_sum(), xa::ptr(reg_prev_out_addr, static_cast<int32_t>((ofs - prev_ofs)>>6)));
            }else{
              if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
                ofs = ofs - prev_ofs;
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, reg_prev_out_addr, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }
              }else{
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, aux_reg_output_data, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }
              }
    
              CGA64::ldr(vreg_sum(), xa::ptr(reg_prev_out_addr));
    
              prev_ofs = jcp.typesize_out * jcp.load_block * i_ur;
            }
          }
        }
      }
      return prev_ofs;

    };

    auto out_str = [=](int i_load, int i_ur, int prev_ofs){
      int ofs;
      if (one_of(jcp.prop_kind, forward_training, forward_inference,
                 backward_data)){
        ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block * jcp.typesize_out;

        if((ofs>>6) < LDRMAX){
          CGA64::str(vreg_accum(i_load, i_ur), xa::ptr(aux_reg_output_data, static_cast<int32_t>(ofs>>6)));
        }else{
          if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&(((ofs - prev_ofs)>>6) < LDRMAX)){
            CGA64::str(vreg_accum(i_load, i_ur), xa::ptr(reg_prev_out_addr, static_cast<int32_t>((ofs-prev_ofs)>>6)));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
              ofs = ofs - prev_ofs;
              if( ofs < ADDMAX){
                CGA64::add( reg_prev_out_addr, reg_prev_out_addr, ofs);
              }else if( ofs < MOVMAX){
                CGA64::mov( reg_tmp_ofs, ofs);
                CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
              }else{
                CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
              }

            }else{
              if( ofs < ADDMAX){
                CGA64::add( reg_prev_out_addr, aux_reg_output_data, ofs);
              }else if( ofs < MOVMAX){
                CGA64::mov( reg_tmp_ofs, ofs);
                CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
              }else{
                CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
              }
            }
            CGA64::str(vreg_accum(i_load, i_ur), xa::ptr(reg_prev_out_addr));
            prev_ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block * jcp.typesize_out;
          }

        }
      }else{
        ofs = jcp.typesize_out * jcp.load_block * i_ur;
        if( i_load ){
          CGA64::mov(reg_tmp_out_ofs, i_load);
          if((ofs>>6) < LDRMAX){
            CGA64::madd(reg_tmp_out_ofs, reg_tmp_out_ofs, reg_output_stride, aux_reg_output_data);
            CGA64::str(vreg_sum(), xa::ptr(reg_tmp_out_ofs, static_cast<int32_t>(ofs>>6)));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&(((ofs - prev_ofs)>>6) < LDRMAX)){
              CGA64::madd(reg_tmp_out_ofs, reg_tmp_out_ofs, reg_output_stride, reg_prev_out_addr);
              CGA64::str(vreg_sum(), xa::ptr(reg_tmp_out_ofs, static_cast<int32_t>((ofs - prev_ofs)>>6)));
            }else{
              if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
                ofs = ofs - prev_ofs;
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, reg_prev_out_addr, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }
              }else{
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, aux_reg_output_data, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }
              }
              CGA64::madd(reg_tmp_out_ofs, reg_tmp_out_ofs, reg_output_stride, reg_prev_out_addr);
              CGA64::str(vreg_sum(), xa::ptr(reg_tmp_out_ofs));
    
              prev_ofs = jcp.typesize_out * jcp.load_block * i_ur;
            }
          }
        }else{
          if((ofs>>6) < LDRMAX){
            CGA64::str(vreg_sum(), xa::ptr(aux_reg_output_data, static_cast<int32_t>(ofs>>6)));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0) &&(((ofs - prev_ofs)>>6) < LDRMAX)){
              CGA64::str(vreg_sum(), xa::ptr(reg_prev_out_addr, static_cast<int32_t>((ofs - prev_ofs)>>6)));
            }else{
              if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
                ofs = ofs - prev_ofs;
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, reg_prev_out_addr, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, reg_prev_out_addr, reg_tmp_ofs);
                }
              }else{
                if( ofs < ADDMAX){
                  CGA64::add(reg_prev_out_addr, aux_reg_output_data, ofs);
                }else if( ofs < MOVMAX){
                  CGA64::mov( reg_tmp_ofs, ofs);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }else{
                  CGA64::mov( reg_tmp_ofs, ofs & 0xffff);
                  CGA64::movk( reg_tmp_ofs, ofs >> 16, 16);
                  CGA64::add( reg_prev_out_addr, aux_reg_output_data, reg_tmp_ofs);
                }
              }
    
              CGA64::str(vreg_sum(), xa::ptr(reg_prev_out_addr));
    
              prev_ofs = jcp.typesize_out * jcp.load_block * i_ur;
            }
          }
        }

      }

      return prev_ofs;
    };

    auto store = [=]() {

        xa::LabelAArch64 store_noadd;
        if (!jcp.with_sum) {
            CGA64::tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            CGA64::b(xa::NE, store_noadd);
        }

        int prev_ofs = -1;
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum_s(i_load, i_ur);
                prev_ofs = out_load(i_load, i_ur, prev_ofs);       
                CGA64::fadd(r, r, vreg_sum_s());
            }

        CGA64::L_aarch64(store_noadd);
        if (jcp.with_eltwise) {
            //assert(!jcp.with_eltwise);
#if 1
            xa::LabelAArch64 store_noeltwise;
            CGA64::cmp(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
            CGA64::b(xa::NE, store_noeltwise);

            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

            CGA64::L_aarch64(store_noeltwise);
#endif 
        }

        auto store_output = [=](bool output_is_aligned) {
        
            int prev_ofs = -1;
            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load){
                    prev_ofs = out_str(i_load, i_ur, prev_ofs);
                }
        };

        xa::LabelAArch64 unaligned_store, end_store;
        CGA64::tst(aux_reg_output_data, cpu_isa_traits<sve>::vlen - 1);
        CGA64::b(xa::NE, unaligned_store);
        store_output(true);
        CGA64::b(end_store);
        CGA64::L_aarch64(unaligned_store); {
            store_output(false);
        }
        CGA64::L_aarch64(end_store);
    };


    auto fma_block = [=](bool last_block) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);

        int reduce_step = jcp.fma_step;
        int prev_bcast_ofs = -1;

        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll;
                i_reduce += reduce_step) { // IC
            int load_scale = 1;
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) { // OC
                // if transposed input data used and if spatial size is
                // not divided by transpose step (4) then for last reduce step
                // we should load only needed load_registers data
                // and clear remaining
                if (jcp.transpose_src && jcp.is % jcp.fma_step && last_block
                        && i_reduce == jcp.reduce_loop_unroll - reduce_step) {
                    xa::LabelAArch64 load_all;
                    xa::LabelAArch64 load_finish;
                    tst(reg_reduce_pos_flag, FLAG_SP_LAST);
                    CGA64::b(xa::EQ, load_all);

                    const int n_loads = jcp.is % jcp.fma_step;
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        if (i_fma < n_loads){
                            load_load(i_reduce + load_scale * i_fma, i_load, i_fma);
                        }else
                            CGA64::fmov(vreg_load_s(i_load, i_fma));
                    }
                    CGA64::b(load_finish);

                    CGA64::L_aarch64(load_all);
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        load_load(i_reduce + load_scale * i_fma, i_load, i_fma);
                    }
                    CGA64::L_aarch64(load_finish);
                } else {
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        load_load(i_reduce + load_scale * i_fma, i_load, i_fma);
                    }
                }
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) { // HW
                prev_bcast_ofs = bcast_load(i_reduce, i_ur, prev_bcast_ofs);

                for (int i_load = 0; i_load < load_loop_blk; ++i_load) { // OC
                    fmla(vreg_accum_s(i_load, i_ur), reg_p_all_ones,
                                vreg_load_s(i_load, 0), vreg_bcast_s());
                }
            }
        }
    };


    xa::LabelAArch64 reduce_loop;
    xa::LabelAArch64 reduce_loop_tail;

    CGA64::mov(aux_reg_load_data, reg_load_data);

    CGA64::mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    CGA64::mov(reduce_loop_iter, reg_reduce_loop_work);
    assert(jcp.reduce_loop_unroll < 4096);
    CGA64::subs(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll);
    CGA64::b(xa::LE, reduce_loop_tail);


    /* Input channel loop */
    CGA64::L_aarch64(reduce_loop); { 
        fma_block(false);

        add_imm(aux_reg_bcast_data, aux_reg_bcast_data, jcp.reduce_loop_bcast_step);

        add_imm(aux_reg_load_data, aux_reg_load_data, jcp.reduce_loop_load_step);
        assert(jcp.reduce_loop_unroll < 4096);
        CGA64::subs(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll);
        CGA64::b(xa::GT, reduce_loop);
    }

    CGA64::L_aarch64(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_sve_1x1_conv_kernel::generate()
{
    preamble();

    ptrue( reg_p_all_ones.b );
    /* Pointers indicate weight, input, and output data */
    CGA64::ldr(reg_bcast_data,   xa::ptr(abi_param1_aarch64, GET_OFF(bcast_data)));    // Input
    CGA64::ldr(reg_load_data,    xa::ptr(abi_param1_aarch64, GET_OFF(load_data)));     // Weight
    CGA64::ldr(reg_output_data,  xa::ptr(abi_param1_aarch64, GET_OFF(output_data)));   // Output

    /*TODO: Check */
#if 0
    CGA64::sub(rsp, rsp, stack_space_needed);
#endif

    /* Pointer indicates bias data if the layer has bias option */
    if (jcp.with_bias)
        CGA64::ldr(reg_bias_data, xa::ptr(abi_param1_aarch64, GET_OFF(bias_data)));

    /* Get workloads of each loop */
    CGA64::ldr(reg_load_loop_work, xa::ptr(abi_param1_aarch64, GET_OFF(load_dim)));
    CGA64::ldr(reg_bcast_loop_work, xa::ptr(abi_param1_aarch64, GET_OFF(bcast_dim)));
    // CGA64::mov(EVEX_compress_addr(rsp, bcast_loop_work_offt), reg_bcast_loop_work); // TODO: necessary?
    CGA64::ldr(reg_reduce_loop_work, xa::ptr(abi_param1_aarch64, GET_OFF(reduce_dim)));

    /* A flag for controlling reduce loop */
    CGA64::ldr(reg_reduce_pos_flag, xa::ptr(abi_param1_aarch64, GET_OFF(first_last_flag)));

    if (one_of(jcp.prop_kind, forward_training, forward_inference))
        CGA64::mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.eltwise.alpha));

    if (jcp.prop_kind == backward_weights)
        CGA64::ldr(reg_output_stride, xa::ptr(abi_param1_aarch64, GET_OFF(output_stride)));

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);

        /* Calculate weight address for next bcast_loop */
        // Added #of load_loops * #of steps in each load_loops
        add_imm(reg_load_data, reg_load_data, load_loop_blk * jcp.load_loop_load_step);

        switch (jcp.prop_kind) {
          case forward_training:
          case forward_inference:

              add_imm(reg_bias_data, reg_bias_data, load_loop_blk * jcp.load_block * jcp.typesize_out);

              add_imm(reg_output_data, reg_output_data, load_loop_blk * jcp.bcast_dim * jcp.load_block *jcp.typesize_out);
              break;
          case backward_data:
              add_imm(reg_output_data, reg_output_data, 
                        load_loop_blk * jcp.bcast_dim * jcp.load_block * jcp.typesize_out);
              break;
          case backward_weights:
              for (int i_load = 0; i_load < load_loop_blk; i_load++){
                  CGA64::add(reg_output_data, reg_output_data, reg_output_stride);
              }
              break;
          default:
              assert(!"invalid prop_kind");
        }
        CGA64::sub(reg_load_loop_work, reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };


    const int simd_w = 16; // The lenght of vector instructions (512-bit)

    xa::LabelAArch64 load_loop_blk[7];

    // # of unrolling in the OC field ??
    static const int ur_cases_bcast[] = { 2, 5, 6, 9, 14, 32 };
    //static const int ur_cases_bcast[] = { 2, 5 };

    const int size_ur_cases = sizeof(ur_cases_bcast);
    const int *ur_cases = ur_cases_bcast;
    const int num_ur_cases = size_ur_cases / sizeof(*ur_cases);

    
    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            assert( (simd_w * (label_idx + 1)) < 4096 );
            CGA64::cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            CGA64::b(xa::LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            CGA64::L_aarch64(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    CGA64::cmp(reg_load_loop_work, 0);
                    CGA64::b(xa::EQ, load_loop_blk[num_ur_cases]);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    CGA64::cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    CGA64::b(xa::EQ, load_loop_blk[label_idx - 1]);
                }
                CGA64::cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                CGA64::b(xa::GE, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                CGA64::cmp(reg_load_loop_work, simd_w * (idx + 1));
                CGA64::b(xa::EQ, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                CGA64::cmp(reg_load_loop_work, simd_w);
                CGA64::b(xa::LE, load_loop_blk[0]);
            }
        }
    }
    CGA64::L_aarch64(load_loop_blk[num_ur_cases]);

#if 0
    CGA64::add(rsp, rsp, stack_space_needed);
#endif 


#if 1
    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
#endif

    postamble();

}

bool jit_sve_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
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

status_t jit_sve_1x1_conv_kernel::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    if (!mayiuse(sve)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = cpu_isa_traits<sve>::vlen / sizeof(float);
    const int ndims = src_d.ndims();
    /* Forward_[training, inference], backward_[data, weight] */
    jcp.prop_kind = cd.prop_kind; 

    // TODO: impl backward_data
    if(!one_of(jcp.prop_kind, forward_training, forward_inference, backward_data)){
      return status::unimplemented;
    }
    /* Check group option: if true, NCHW -> gNCHW */
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;

    /* Batchsize */
    jcp.mb = src_d.dims()[0];

    /* Channel */
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups; // output channel size
    jcp.ic = src_d.dims()[1] / jcp.ngroups; // input channel size

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    /* HW */
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    /* Kernel HW (==1) */
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    /* Padding */
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];
    
    /* Stride */
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.src_fmt = src_d.format(); // maybe, nChw16c
    /* Bias check */
    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format,
            memory_format::undef, cd.diff_bias_desc.format)
        != memory_format::undef;


    jcp.os = jcp.oh * jcp.ow; 
    jcp.is = jcp.ih * jcp.iw;
    jcp.tr_is = rnd_up(jcp.is, 4);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;


    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
#if __ARM_ARCH
      return status::unimplemented;
#else
      jcp.eltwise = p.entry_[eltwise_ind].eltwise;
      if (dst_d.data_type() == data_type::s32) return status::unimplemented;
#endif
    }

    bool args_ok = true
        && jcp.ngroups == 1
        && everyone_is(pick(ndims - 3, nCw16c, nChw16c), src_d.format(),
            dst_d.format())
        && one_of(cd.bias_desc.format, memory_format::undef, any, x);
    if (!args_ok) return status::unimplemented;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.t_pad == 0 && jcp.l_pad == 0 && jcp.stride_w == 1
            && jcp.stride_h == 1 // TODO: support some strides
            && jcp.ow == jcp.iw && jcp.oh == jcp.ih // enforce rpad=0
            && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok)
        return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;
    jcp.transpose_src = false;

    if (everyone_is(data_type::f32, src_d.data_type(),
                            weights_d.data_type(), dst_d.data_type()))
    {
        const int is_bwd_d = jcp.prop_kind == backward_data;
        memory_format_t weights_format = with_groups
            ? pick(2 * ndims - 6 + is_bwd_d, gOIw16i16o, gIOw16o16i,
                gOIhw16i16o, gIOhw16o16i)
            : pick(2 * ndims - 6 + is_bwd_d, OIw16i16o, IOw16o16i,
                OIhw16i16o, IOhw16o16i);

        if (weights_d.format() != weights_format)
            return status::unimplemented;

        jcp.ver = ver_sve;
        jcp.fma_step = 1;

        jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
        jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);
    } else {
        return status::unimplemented;
    }

    /* once all the formats are set, check the padding consistency */
    args_ok = true
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 28;
    const int BIG_REDUCE_DIM = 1024;
    const int BIG_LOAD_DIM = 256;

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };
    int reduce_blocking_max{ 0 };

    jcp.load_grp_count = 1;

#if 0
    const int L1_capacity = get_cache_size(1, true) / sizeof(float);
    const int L2_size     = get_cache_size(2, true) / sizeof(float);
    const int L2_capacity = (L2_size * 3) / 4;
#else
    const int L1_capacity = 64000 / sizeof(float);
    const int L2_size     = 8000000 / sizeof(float);
    const int L2_capacity = (L2_size*3) / 4;
#endif

    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {

        if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
            jcp.reduce_dim = jcp.ic;
            jcp.reduce_block = jcp.ic_block;

            jcp.load_dim = jcp.oc;
            jcp.load_block = jcp.oc_block;

            jcp.bcast_dim = jcp.is;
        } else {
            jcp.reduce_dim = jcp.oc;
            jcp.reduce_block = jcp.oc_block;

            jcp.load_dim = jcp.ic;
            jcp.load_block = jcp.ic_block;

            jcp.bcast_dim = jcp.os;
        }
        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;
        jcp.load_loop_load_step
                = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

        // adjusting registry blocking
        int max_regs, min_regs, size_treshold, ur_step;
        const int spatial
                = (one_of(jcp.prop_kind, forward_training, forward_inference)) ?
                jcp.oh :
                jcp.ih;

#ifdef __ARM_ARCH
        max_regs = 9;
        min_regs = 6;
        size_treshold = 14;
        ur_step = 1;
        jcp.expl_bcast = true;

        if (jcp.load_dim > 128 && jcp.load_dim < BIG_LOAD_DIM
                && spatial > SMALL_SPATIAL && spatial < BIG_SPATIAL) {
            max_regs = 6;
            min_regs = 5;
        }
#else
        if (jcp.ver == ver_avx512_core && (8 * jcp.mb) / nthreads >= 1) {
            max_regs = 9;
            min_regs = 6;
            size_treshold = 14;
            ur_step = 1;
            jcp.expl_bcast = true;

            if (jcp.load_dim > 128 && jcp.load_dim < BIG_LOAD_DIM
                    && spatial > SMALL_SPATIAL && spatial < BIG_SPATIAL) {
                max_regs = 6;
                min_regs = 5;
            }
        } else {
            max_regs = 30;
            min_regs = 9;
            size_treshold = 14;
            ur_step = 1;
            jcp.expl_bcast = false;
            jcp.use_vmovntps = true;
        }
#endif // __ARM_ARCH

        jcp.ur = 1;

        for (int ur_w = max_regs; ur_w >= min_regs; ur_w -= ur_step) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i -= ur_step) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0)
                        break;
                }
            }
        }

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        jcp.bcast_block = jcp.ur;

        jcp.bcast_loop_output_step = jcp.ur * jcp.load_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.reduce_block * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_iter_step = jcp.load_block;

        if (jcp.prop_kind == backward_data)
            jcp.loop_order = loop_lbr;
        else
            jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);
        int nb_load = div_up(jcp.load_dim, jcp.load_block);

        if (jcp.ver == ver_sve && jcp.expl_bcast) {
            if (jcp.load_dim <= BIG_LOAD_DIM && spatial > SMALL_SPATIAL
                    && spatial < BIG_SPATIAL)
                reduce_blocking = nstl::min(jcp.reduce_dim, 80);
            else if (spatial > SMALL_SPATIAL)
                reduce_blocking = nstl::min(jcp.reduce_dim, 512);
            else
                reduce_blocking = nstl::min(jcp.reduce_dim, 256);

            if ((jcp.mb > 28 && spatial >= 28)
                    || (jcp.mb > 112 && spatial >= 17))
                jcp.use_vmovntps = true;
            else
                jcp.use_vmovntps = false;
        } else {

            reduce_blocking = nb_reduce;
            if (spatial <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 16;
            else if (spatial > SMALL_SPATIAL
                    && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 8;
//            reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        // Check input data cache aliasing.
        // For other ISA constants may be updated.
        // 64 * 1024 is chosen due to 1MB L2 16-way cache.
        // 7 is empirical value. It is about half of 16.
        // So we leave about half of the set for other data - weights, dst
        int way_size = (64 * 1024) / jcp.typesize_in;
        int max_hits = 7;
        if (jcp.bcast_dim * reduce_blocking > way_size * max_hits) {
            int nrb = reduce_blocking / simd_w;
            int sp = jcp.bcast_dim;
            int wl = way_size / simd_w;
            for (int start_off = 0; start_off < jcp.ur; start_off++) {
                for (int off = start_off, hits = 0; off < sp * nrb; off += wl) {
                    if (off % sp >= jcp.ur || ++hits < max_hits)
                        continue;
                    int max_r_blocking = simd_w * nstl::max(1, (off + wl) / sp);
                    reduce_blocking
                            = nstl::min(reduce_blocking, max_r_blocking);
                    break;
                }
            }
        }

        if (reduce_blocking < jcp.reduce_dim) {
            jcp.use_vmovntps = false;
            if (jcp.prop_kind == backward_data)
                jcp.loop_order = reduce_src ? loop_lbr : loop_rlb;
            else
                jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
        }
        load_blocking = jcp.load_dim;

        int load_size = jcp.load_dim * jcp.reduce_dim;
        int bcast_size = jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;

        if (jcp.ver == ver_sve && nthreads <= 28 && jcp.mb < nthreads
                && nb_load * nb_bcast > nthreads) {
            // Some heuristic here
            float calc_koef = 0.01, best_cost = FLT_MAX;
            int n_lgc = nthreads;
            float ratio = (float)load_size / (float)bcast_size;
            int best_lgc = ratio > 1 ? n_lgc : 1;
            auto calc_job_cost = [&](int lb, int tg, float mem_k) {
                int bb_size = jcp.mb * div_up(nb_bcast, tg);
                float calc_size = (float)(bb_size * jcp.ur)
                        * (lb * jcp.load_block) * jcp.reduce_dim;
                float mem_size = (float)(bb_size * jcp.ur + lb * jcp.load_block)
                        * jcp.reduce_dim;
                return calc_koef * calc_size + mem_k * mem_size;
            };
            for (int lgc, ilgc = 0; ilgc < n_lgc; ilgc++) {
                lgc = ratio > 1 ? n_lgc - ilgc : ilgc + 1;
                int min_lb = nb_load / lgc;
                int max_lb = div_up(nb_load, lgc);
                int min_tg = nthreads / lgc;
                int max_tg = div_up(nthreads, lgc);
                // Some heuristic here
                float mem_koef = (max_tg == 1) ? 1.f : 1.3f;
                float job_cost = 0.;
                if (nthreads % lgc < nb_load % lgc) {
                    job_cost = calc_job_cost(max_lb, min_tg, mem_koef);
                } else {
                    auto job_cost1 = calc_job_cost(max_lb, max_tg, mem_koef);
                    auto job_cost2 = calc_job_cost(min_lb, min_tg, mem_koef);
                    job_cost = nstl::max(job_cost1, job_cost2);
                }

                if (job_cost < best_cost) {
                    best_lgc = lgc;
                    best_cost = job_cost;
                }
            }
            jcp.load_grp_count = best_lgc;
            load_blocking = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        } else {
            jcp.load_grp_count = div_up(nthreads, jcp.mb * jcp.ngroups * nb_bcast);
//            jcp.load_grp_count = best_divider(
//                nthreads, jcp.load_grp_count, 2 * jcp.load_grp_count, false);
        }

        if (jcp.ver == ver_sve && jcp.expl_bcast && jcp.bcast_dim <= 64
                && load_size >= L2_size) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
        } else if (jcp.bcast_dim <= 49 && jcp.mb <= nthreads
                && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                 div_up(nthreads, jcp.load_grp_count))
                * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast
                = (L2_capacity - /* kernel_size - */
                    2 * jcp.load_block * reduce_blocking
                        - jcp.ur * reduce_blocking - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity)
            space_for_bcast /= 2;

        int bcast_in_cache
                = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

#if 1
    } else if (jcp.prop_kind == backward_weights) {

        jcp.use_vmovntps = false;

        if (jcp.transpose_src)
            jcp.reduce_dim = jcp.tr_is;
        else
            jcp.reduce_dim = jcp.is;

        jcp.reduce_block = best_divider(jcp.reduce_dim, 7, 16, true);
        if (jcp.reduce_dim % jcp.reduce_block != 0)
            jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false);
        if (jcp.reduce_block > 256) {
            jcp.reduce_block = 1;
        }

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        if (jcp.ver == ver_avx512_core && jcp.reduce_block <= 19) {
            // if reduce_block is big then generated JIT code may be big
            // for small values of ur because reduce_loop_unroll = reduce_block
            jcp.ur = jcp.bcast_block / 2;
            jcp.expl_bcast = true;
        } else {
            jcp.ur = jcp.bcast_block;
            jcp.expl_bcast = false;
        }

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.ic_block * jcp.typesize_in;
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * jcp.typesize_in;

        jcp.bcast_loop_output_step =
                                jcp.oc_block * jcp.ic_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep =
            jcp.oc_block * jcp.ur * jcp.typesize_out;
        jcp.bcast_loop_bcast_step =
                jcp.ic_block * jcp.reduce_dim * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = jcp.ur * jcp.typesize_in;

        jcp.load_loop_load_step = jcp.oc_block * jcp.os * jcp.typesize_in;
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */
        balance(jcp, nthreads);

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        load_blocking = best_divider(load_blocking, 16, load_blocking, false);
        load_blocking *= jcp.load_block;

        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        int max_bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        int min_bcast_blocking = 5;

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        bcast_blocking = best_divider(
                bcast_blocking, min_bcast_blocking, max_bcast_blocking, false);
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        // for reduction balance
        if (jcp.ver == ver_avx512_core) {
            int max_reduce_blocking
                    = nstl::min(L1_capacity / jcp.ur, jcp.reduce_dim);
            int min_reduce_blocking = nstl::min(
                    L1_capacity / jcp.ur, nstl::max(jcp.iw, jcp.ih));
            reduce_blocking = best_divider(jcp.reduce_dim, min_reduce_blocking,
                    max_reduce_blocking, true);
            reduce_blocking
                    = nstl::max(rnd_dn(reduce_blocking, jcp.reduce_block),
                            jcp.reduce_block);
        } else {
            int max_reduce_blocking = L2_capacity
                    / ((bcast_blocking + load_blocking) * jcp.reduce_block);
            max_reduce_blocking = nstl::min(max_reduce_blocking,
                    (L1_capacity / (jcp.bcast_block)) / jcp.reduce_block);

            int num_jobs = div_up(jcp.load_dim, load_blocking)
                    * div_up(jcp.bcast_dim, bcast_blocking);
            int threads_per_job = nstl::max(1, nthreads / num_jobs);
            reduce_blocking = div_up(jcp.mb * jcp.reduce_dim, jcp.reduce_block);
            reduce_blocking = div_up(reduce_blocking, threads_per_job);

            reduce_blocking = best_divider(reduce_blocking,
                    max_reduce_blocking - 2, max_reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
#endif //#if 0
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

#if 0
    std::cout << "jit_sve_check: success" << std::endl; // honda
    std::cout << "#weight: " << weights_d.ndims() << " " << weights_d.dims()[0] << " " << weights_d.dims()[1] << " " << weights_d.dims()[2] << std::endl;
    std::cout << "#src: " << src_d.ndims() << " " << src_d.dims()[0] << " " << src_d.dims()[1] << " " << src_d.dims()[2] << std::endl;
    std::cout << "#dst: " << dst_d.ndims() << " " << dst_d.dims()[0] << " " << dst_d.dims()[1] << " " << dst_d.dims()[2] << std::endl;
    std::cout << "#jcp: " << jcp.mb << " " << jcp.ic << " " << jcp.oc << " " << jcp.ih << " " << jcp.iw << " " << jcp.oh << " " << jcp.ow << std::endl;
#endif
    return status::success;
}

void jit_sve_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {
    using namespace mkldnn::impl::memory_tracking::names;
    if (jcp.prop_kind != backward_data && jcp.with_bias
            && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.typesize_out * jcp.oc);

    if (jcp.prop_kind == backward_weights) {
        const size_t wei_size = (size_t)jcp.ngroups * jcp.oc * jcp.ic;
        scratchpad.book(key_conv_wei_reduction,
                jcp.typesize_out * wei_size * (jcp.nthr_mb - 1));
    }

    if (jcp.transpose_src) {
        const size_t tr_src_size =
            (size_t)jcp.nthr_mb * jcp.ngroups * jcp.ic * jcp.tr_is;
        scratchpad.book(key_conv_tr_src, jcp.typesize_out * tr_src_size);
        scratchpad.book(key_conv_tr_src_bctx,
                sizeof(simple_barrier::ctx_t) * jcp.nthr);
    }

}

void jit_sve_1x1_conv_kernel::balance(jit_1x1_conv_conf_t &jcp,
        int nthreads)
{
    // initialize jcp reduction threading properties
    jcp.nthr = jcp.nthr_mb = jcp.nthr_g = jcp.nthr_oc_b = jcp.nthr_ic_b = 1;
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }
    const int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    const int nb_load = div_up(jcp.load_dim, jcp.load_block);
    const int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    jcp.nthr_g = jcp.ngroups;
    const int nthr = nthreads / jcp.nthr_g;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
        * optimizer tries to minimize memory consumption. few notes: (n1)
        * unclear why, but that essentially helps first convolution...
        *  (n2) assuming the reduction over minibatch is always there:
        *    - instead of 8 it should be 5 here (write ~= 2 read):
        *      kernel: temporal workspace 1 write
        *      reduction: 1 read from workspace and 1 write to the diff_wei
        *    - but experiments showed 8 works better than 5 or 6... */
        int bcast_koeff = 1;
        int load_koeff = 1;
        int output_koeff = 12;
        if (jcp.transpose_src) {
            bcast_koeff = 5;
            load_koeff = 1;
            output_koeff = 8;
        }
        return 0
            + (size_t)bcast_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
            * div_up(jcp.ngroups, jcp.nthr_g)
            * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.reduce_block
            / jcp.stride_h / jcp.stride_w /* (n1) */
            + (size_t)load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
            * div_up(jcp.ngroups, jcp.nthr_g)
            * div_up(nb_load, nthr_oc_b) * jcp.oc_block * jcp.reduce_block
            + (size_t)output_koeff /* (n2) */
            * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
            * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block
            * jcp.oc_block;
    };

    int nthr_mb = 1, nthr_oc_b = 1, nthr_ic_b = 1;
    auto best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, jcp.mb * nb_reduce);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, nb_load);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, nb_bcast);
            auto mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                jcp.nthr_mb = nthr_mb;
                jcp.nthr_oc_b = nthr_oc_b;
                jcp.nthr_ic_b = nthr_ic_b;
            }
        }

        if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
    }
    if (jcp.nthr_mb > nthreads / 2 && jcp.nthr_mb < nthreads)
        jcp.nthr_mb = nstl::min(jcp.mb, nthreads);

    jcp.nthr = jcp.nthr_mb * jcp.nthr_g * jcp.nthr_oc_b * jcp.nthr_ic_b;
    assert(jcp.nthr <= nthreads);
}

}
}
}
