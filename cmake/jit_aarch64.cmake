#===============================================================================
# Copyright 2020 FUJITSU LIMITED
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

# Manage different library options
#===============================================================================

if(jit_aarch64_cmake_included)
    return()
endif()
set(jit_aarch64_cmake_included true)

# =============================
# JIT type selection
# =============================
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR STREQUAL "")
    message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR is not defined. Perhaps CMake toolchain is broken")
else()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    option(DNNL_NATIVE_JIT_AARCH64
        "enables native JIT for AArch64."
        ON) # disabled by default on AArch64 CPU
else()
    option(DNNL_NATIVE_JIT_AARCH64
        "enables native JIT for AArch64."
        OFF) # disabled by default on x86_64 CPU
endif()

message(STATUS "DNNL_NATIVE_JIT_AARCH64=${DNNL_NATIVE_JIT_AARCH64}")
