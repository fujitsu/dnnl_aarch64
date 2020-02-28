# Build instruction

```
git clone http://kaiseki-juku.parc.flab.fujitsu.co.jp/postk_dl/dnnl_aarch64.git
cd dnnl_aarch64/
git checkout fjdev
git submodule init
git submodule update
cd third_party/
mkdir build_xed_aarch64
cd build_xed_aarch64/

cat ~/build_xed.sh
> #!/bin/bash
> #PJM -L "rscunit=rscunit_ft01,rscgrp=def_grp"
> #PJM -L elapse=00:10:00
> #PJM -L "node=1"
> #PJM -S
> 
> echo ${0}
> ../xed/mfile.py --shared examples install

pjsub ~/build_xed.sh
cd kits/
ln -sf xed-install-base-2020-01-30-lin-x86-64 xed
cd ../../../
mkdir build_aarch64
cd build_aarch64/

cat ~/cmake_native_jit_reorder_debug.sh
> #!/bin/bash
> ##PJM -L "rscunit=rscunit_ft01,rscgrp=interact"
> #PJM -L "rscunit=rscunit_ft01,rscgrp=def_grp"
> #PJM -L elapse=00:02:00
> #PJM -L "node=1"
> #PJM -S
> 
> echo ${0}
> /fefs01/ai/Kawakami/local_a64fx/bin/cmake .. -DCMAKE_BUILD_TYPE=Debug -DXBYAK_TRANSLATE_AARCH64=ON -DDNNL_AARCH64_NATIVE_JIT_REORDER=ON -DXBYAK_XED_LIB_ARCH_IS_AARCH64=ON

pjsub ~/cmake_native_jit_reorder_debug.sh

cat ~/make.sh
> #!/bin/bash
> #PJM -L "rscunit=rscunit_ft01,rscgrp=def_grp"
> #PJM -L elapse=00:10:00
> #PJM -L "node=1"
> #PJM -S
> 
> echo ${0}
> make -j40

pjsub ~/make.sh
cd tests/gtests

cat ~/test_reorder.sh
> #!/bin/bash
> #PJM -L "rscunit=rscunit_ft01,rscgrp=def_grp"
> #PJM -L elapse=00:10:00
> #PJM -L "node=1"
> #PJM -S
> 
> echo ${0}
> export MKLDNN_VERBOSE=1
> export MKLDNN_JIT_DUMP=1
> ./test_reorder

pjsub ~/test_reorder.sh
