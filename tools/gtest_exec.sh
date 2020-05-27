#!/bin/bash
#===============================================================================
# Copyright 2019-2020 FUJITSU LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
list_bin="test_batch_normalization_f32 test_batch_normalization_s8 test_concat test_convolution_backward_data_f32 test_convolution_backward_data_s16s16s32 test_convolution_backward_weights_f32 test_convolution_backward_weights_s16s16s32 test_convolution_eltwise_forward_f32 test_convolution_eltwise_forward_x8s8f32s32 test_convolution_format_any test_convolution_forward_f32 test_convolution_forward_s16s16s32 test_convolution_forward_u8s8fp test_convolution_forward_u8s8s32 test_deconvolution test_eltwise test_gemm_f32 test_iface_attr test_iface_pd_iter test_inner_product_backward_data test_inner_product_backward_weights test_inner_product_forward test_lrn_backward test_lrn_forward test_memory test_mkldnn_threading test_pooling_backward test_pooling_forward test_reorder test_rnn_forward test_shuffle test_softmax_backward test_softmax_forward test_sum" 

for i in ${list_bin} ; do
    echo ${i} 
    ./${i}.sh | tee ${i}.log
done
