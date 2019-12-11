#!/bin/bash
# This script modify cmake output files under cross compile environment.
#
# If you want to build mkldnn for a64fx and run on x64 linux,
# do the following commands.
# export CC=/usr/bin/aarch64-linux-gnu-gcc
# export CXX=/usr/bin/aarch64-linux-gnu-g++
# export LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib
# cd mkldnn_for_a64fx/mkl-dnn
# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# mkldnn_for_a64fx/tools/replace.sh
# make -j28
# cd tests/gtests
# ./test_reorder


list=`find . \( -name "flags.make" -o -name "link.txt" \)`

for i in ${list} ; do
    cat ${i} | sed -e "s/\-march\=native/\-march\=armv8-a/" | sed -e "s/\-mtune\=native//" > hogefugafuga
    mv hogefugafuga ${i}
done
