# Build instruction

```
git clone http://kaiseki-juku.parc.flab.fujitsu.co.jp/postk_dl/dnnl_aarch64.git
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

