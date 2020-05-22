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

# - Find BLAS library
# This module finds an installed fortran library that implements the BLAS
# linear-algebra interface (see http://www.netlib.org/blas/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  BLAS_FOUND - set to true if a library implementing the BLAS interface is found.
#  BLAS_INFO - name of the detected BLAS library.
#  BLAS_F2C - set to true if following the f2c return convention
#  BLAS_LIBRARIES - list of libraries to link against to use BLAS
#  BLAS_INCLUDE_DIR - include directory

# Do nothing if BLAS was found before
IF(MKLDNN_USE_MKL STREQUAL "NONE")

SET(BLAS_LIBRARIES)
SET(BLAS_INCLUDE_DIR)
SET(BLAS_INFO)


# Old FindBlas
INCLUDE(CheckCSourceRuns)
INCLUDE(CheckFortranFunctionExists)

MACRO(Check_Fortran_Libraries LIBRARIES _prefix _name _flags _list)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to NOTFOUND.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.

  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  message(STATUS "Checking for [${__list}]")

  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      if ( WIN32 )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS ENV LIB
          PATHS ENV PATH )
      endif ( WIN32 )
      if ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 /opt/OpenBLAS/lib /usr/lib/aarch64-linux-gnu
          ENV DYLD_LIBRARY_PATH )
      else ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 /opt/OpenBLAS/lib /usr/lib/aarch64-linux-gnu
          ENV LD_LIBRARY_PATH )
      endif( APPLE )
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      MESSAGE(STATUS "  Library ${_library}: ${${_prefix}_${_library}_LIBRARY}")
    endif(_libraries_work)
  endforeach(_library ${_list})
  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
    if (CMAKE_Fortran_COMPILER_WORKS)
      check_fortran_function_exists(${_name} ${_prefix}${_combined_name}_WORKS)
    else (CMAKE_Fortran_COMPILER_WORKS)
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif (CMAKE_Fortran_COMPILER_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    mark_as_advanced(${_prefix}${_combined_name}_WORKS)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif(_libraries_work)
  if(NOT _libraries_work)
    set(${LIBRARIES} NOTFOUND)
  endif(NOT _libraries_work)
endmacro(Check_Fortran_Libraries)

MACRO(Check_Blas_Libraries LIBRARIES _prefix _name _flags _list)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to NOTFOUND.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.

  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  message(STATUS "Checking for [${__list}]")

  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      if ( WIN32 )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS ENV LIB
          PATHS ENV PATH )
      endif ( WIN32 )
      if ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 /opt/OpenBLAS/lib /usr/lib/aarch64-linux-gnu
          ENV DYLD_LIBRARY_PATH )
      else ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 /opt/OpenBLAS/lib /usr/lib/aarch64-linux-gnu
          ENV LD_LIBRARY_PATH )
      endif( APPLE )
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      MESSAGE(STATUS "  Library ${_library}: ${${_prefix}_${_library}_LIBRARY}")
    endif(_libraries_work)
  endforeach(_library ${_list})
  if(NOT _libraries_work)
    set(${LIBRARIES} NOTFOUND)
  endif(NOT _libraries_work)
endmacro(Check_Blas_Libraries)

# BLAS in SSL2 library?
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "ssl2")))
  if (CMAKE_CXX_COMPILER MATCHES ".*/FCC$" AND
      CMAKE_C_COMPILER MATCHES ".*/fcc$")
    check_fortran_libraries(
    BLAS_LIBRARIES
    BLAS
    sgemm
    "-SSL2;--linkfortran"
    "fjlapackexsve")
  endif()
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "ssl2")
    if (CMAKE_CXX_COMPILER MATCHES ".*/FCC$" AND
	CMAKE_C_COMPILER MATCHES ".*/fcc$")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -SSL2 --linkfortran")
    endif()
  endif (BLAS_LIBRARIES)
endif()

# cblas BLAS library?
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "cblas")))
  check_blas_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "cblas")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "cblas")
  endif (BLAS_LIBRARIES)
endif()

# openblas BLAS library?
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "openblas")))
  check_blas_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "openblas")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "openblas")
  endif (BLAS_LIBRARIES)
endif()

# epilogue

if(BLAS_LIBRARIES)
  set(BLAS_FOUND TRUE)
else(BLAS_LIBRARIES)
  set(BLAS_FOUND FALSE)
endif(BLAS_LIBRARIES)

IF(BLAS_FOUND)
  MESSAGE(STATUS "Found a library with BLAS API (${BLAS_INFO}).")
  list(APPEND EXTRA_SHARED_LIBS ${BLAS_LIBRARIES})
  add_definitions(-DUSE_CBLAS)
  include_directories(AFTER ${BLAS_INCLUDE_DIR})
ELSE(BLAS_FOUND)
  MESSAGE(STATUS "Cannot find a library with BLAS API. Not using BLAS.")
ENDIF(BLAS_FOUND)

# Do nothing is BLAS was found before
ENDIF(MKLDNN_USE_MKL STREQUAL "NONE")
