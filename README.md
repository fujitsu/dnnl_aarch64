# Deep Neural Network Library for AArch64 (DNNL_aarch64)

- An open-source performance library for deep learning applications running on ARM(R)v8-A architecture CPUs
- Optimized to ARMv8-A architecture with the Scalable Vector Extension (SVE)
- The key components are **Xbyak**, **Xbyak_aarch64**, and **Xbyak_Translator**
  - **Xbyak** :  A JIT-assembler for x86 and x64 architectures developed by Shigeo MITSUNARI (Cybozu Labs Inc.)
  - **Xbyak_aarch64** : A JIT-assembler for ARMv8-A architecture of Xbayk
  - **Xbyak_Translator** : A translator which generates JIT functions for ARMv8 with SVE from JIT functions for x86
- Developed based on version 0.21.2 of Deep Neural Network Library (DNNL) by Intel(R)



## Development status

DNNL_aarch64 generates two types of JIT functions for FP32 operations using Xbyak, Xbyak_aarch64, and Xbyak_Translator on ARMv8 with SVE processors

- One is to generate JIT functions for AArch64 directly using Xbyak_aarch64, which is called **Direct method**. The following operations are generated by the method.
  - Convolution
  - Reorder
- The other is a **JIT-translation** from JIT functions for x64 to JIT functions for AArch64 using Xbyak, Xbyak_aarch64, and Xbyak_Translator, which is called **Indirect method**. The following operations are generated by the method.
  - Batch normalization
  - Eltwise
  - Pooling
  - Concat
  - Softmax
  - Sum
  - RNN operations

Reference implementations by C++ run other than those above operations and unsupported parameter sets. They output correct result, but run somewhat slow.

**Bfloat16 support** : Currently, DNNL_aarch64 does not support

### Validated Configurations

| **CPU**      | Fujitsu FX1000 / 700                  |
| ------------ | ------------------------------------- |
| **OS**       | RedHad 8.1 / Centos 8.1               |
| **Compiler** | Fujitsu compiler / GCC 8.3.1 20190507 |



## Requirements

Currently, DNNL_aarch64 is intended to run on CPUs of ARMv8-A with SVE. If you run DNNL_aarch64 on CPUs without SVE, it will be aborted because of undefined instruction exception. 



## Installation

1. Download DNNL_aarch64 from the repository.

```
git clone https://github.com/fujitsu/dnnl_aarch64.git
```

2. Update submodule

```
cd dnnl_aarch64/
git submodule update --init --recursive
```

3. Build xed library

```
mkdir third_party/build_xed_aarch64
pushd third_party/build_xed_aarch64/
../xbyak_translator_aarch64/translator/third_party/xed/mfile.py --shared examples install
cd kits/
ln -sf xed-install-base-* xed
popd
```

4. Build DNNN_aarch64

```
mkdir build_aarch64
cd build_aarch64/
cmake ..
make -j40
```
   - Using BLAS (Optional)

     1. Set the path to the BLAS library on your environment into `LD_LIBRARY_PATH`
     2. Add the following options to cmake command

     | BLAS     | Option                                        |
     | -------- | --------------------------------------------- |
     | SSL2     | -DWITH_BLAS=ssl2 (only with FUJITSU compiler) |
     | openblas | -DWITH_BLAS=openblas                          |

5. Test DNNL_aarch64 (optional)

```
cd tests/gtests
MKLDNN_VERBOSE=1 MKLDNN_JIT_DUMP=1 ./test_reorder
```



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

