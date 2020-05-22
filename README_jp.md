# DNNL_aarch64 ; Deep Neural Network Library for AArch64

## 概要

- Deep Neural Network Libary for AArch64 (DNNL_aarch64) は、ARM(R)v8-A アーキテクチャの CPU 向けのオープンソースライブラリです。
- DNNL_aarch64 は、インテル社の x86-64 CPU 向けに開発された Deep Neural Network Library (DNNL) の、v0.21.2 をベースとしています。
- DNNL_aarch64 は、ARMv8-A アーキテクチャの SVE (Scalable Vector Extension) 命令に最適化されています。
- DNNL_aarch64 は、X86-64 向け Xbyak をARMv8-Aへ拡張した Xbyak_aarch64と、x86_64 向けに記述された Xbyak コードから ARMv8-A 命令に置き換えて実行します。
　Xbayk は、サイボウズ・ラボの光成氏によって開発されたオープンソースソフトウェアです。


## Development status

Intel's original DNNL has varios operation kernels of deep learning, such as 
batch_normalization, concat, convolution, deconvolution, eltwise, pooling, reorder, 
rnn_forward, shuffle, softmax, sum.
Intel has tuned these operation kernels by using JIT assembler 'Xbyak'.
We are applying same technique to DNNL_aarch64 with 'Xbyak_aarch64' in order to improve 
performance of operation kernels running on ARMv8-A CPUs,
but only reorder kernel has been JIT-ed. The other kernels are currently under development.
If you use the other kernels of DNNL_aarch64, reference C++ implementation is used. They 
output correct result, but run somewhat slow.
DNNL includes the following example and test programs, which are built in 
"YOUR_BUILD_DIRECTORY/examples" and "YOUR_BUILD_DIRECTORY/tests/gtests" directories.
DNNL_aarch64 also contains the same programs. You can run them on ARMv8-a enviroment.
You can execute DNNL_aarch64 on systems using Arm(R)v8-A architecure CPUs supporting SVE 
instructions.
Even if you can't access such systems, you can try DNNL_aarch64 on QEMU (generic and open 
source machine emulator and virtualizer).

動作確認済み環境
 Fujitsu FX1000/FX700
 RedHat 8.1 / Centos 8.1
 gcc
 
オプション
 OpenBLAS
 富士通コンパイラFJSV


## Requirements

- Currently, DNNL_aarch64 is intended to run on CPUs of ARMv8-A with SVE. If you run 
DNNL_aarch64 on CPUs without SVE, it will be aborted because of undefined instruction 
exception. 


## Installation

Download DNNL_aarch64 source code or clone the repository.

```
git clone https://github.com/fujitsu/dnnl_aarch64.git

cd dnnl_aarch64/
git checkout fjdev
git submodule update --init --recursive
(or pushd third_party/xbyak; git submodule update -i)

cd third_party/
mkdir build_xed_aarch64
cd build_xed_aarch64/
../xbyak/translator/third_party/xed/mfile.py --shared examples install
cd kits/
ln -sf xed-install-base-* xed
cd ../../../

mkdir build_aarch64
cd build_aarch64/
cmake .. -DCMAKE_BUILD_TYPE=Debug -DXBYAK_TRANSLATE_AARCH64=ON -DDNNL_AARCH64_NATIVE_JIT_REORDER=ON -DXBYAK_XED_LIB_ARCH_IS_AARCH64=ON
make -j40

cd tests/gtests
MKLDNN_VERBOSE=1 MKLDNN_JIT_DUMP=1 ./test_reorder
```


Then, please follow the installation instruction written in [README_intel.md](README_intel.md).


## License

Copyright FUJITSU LIMITED 2019-2020

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Notice

* Arm is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.
* Intel is a registered trademark of Intel Corporation (or its subsidiaries) in the US and/or elsewhere.

## History

|Date|Version|Remarks|
|----|----|----|
|December 11, 2019|0.9.0_base_0.19|First public release version.|
|May 31, 2020|1.0.0_base_0.21.2|Update|


## Copyright

Copyright FUJITSU LIMITED 2019-2020
