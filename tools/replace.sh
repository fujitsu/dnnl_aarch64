#!/bin/bash
# This script modify cmake output files under cross compile environment.
#
# If you want to build dnnl_aarch64 and run on x64 linux,
# do the following commands.
#
# export CC=/usr/bin/aarch64-linux-gnu-gcc
# export CXX=/usr/bin/aarch64-linux-gnu-g++
# export LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib
# cd dnnl_aarch64/mkl-dnn
# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# ../../tools/replace.sh
# make -j28
# cd tests/gtests
# qemu-aarch64 ./test_reorder


list=`find . \( -name "flags.make" -o -name "link.txt" \)`

for i in ${list} ; do
    cat ${i} | sed -e "s/\-march\=native/\-march\=armv8-a/" | sed -e "s/\-mtune\=native//" > hogefugafuga
    mv hogefugafuga ${i}
done
