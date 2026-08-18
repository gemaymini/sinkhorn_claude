# ===========================================================================================================
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ===========================================================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(sinkhorn_normalize LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ============================================================================
# Target 1: 可执行文件 (direct invocation)
# ============================================================================

add_executable(sinkhorn_normalize op_host/sinkhorn_normalize.asc)

target_link_libraries(sinkhorn_normalize PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(sinkhorn_normalize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(sinkhorn_normalize PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ============================================================================
# Target 3: S1 前置 API 探针（独立可执行，不依赖 torch）
# ============================================================================

add_executable(probe_sinkhorn probe/probe_main.asc)

target_link_libraries(probe_sinkhorn PRIVATE
    tiling_api register platform unified_dlog dl m graph_base
)

target_include_directories(probe_sinkhorn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/probe
)

target_compile_options(probe_sinkhorn PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ============================================================================
# Target 2: TORCH_LIBRARY module (libsinkhorn_normalize_ops.so)
# ============================================================================

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

# ---- P0: host 侧优化开关（用于 A/B 与逐基因消融）----
#   SN_TILING_MODE 0=每次分配+同步memcpy(原实现)  1=静态缓存(默认)
#   SN_CORE_QUERY  0=每次查询核数(原实现)         1=首次后缓存(默认)
set(SN_TILING_MODE 1 CACHE STRING "tiling upload mode: 0=per-call memcpy, 1=cached")
set(SN_CORE_QUERY  1 CACHE STRING "core count query: 0=per-call, 1=cached")

add_library(sinkhorn_normalize_ops SHARED
    op_kernel/sinkhorn_normalize_kernel.asc
    op_extension/sinkhorn_normalize_torch.cpp
    op_extension/register.cpp
)

target_include_directories(sinkhorn_normalize_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(sinkhorn_normalize_ops PRIVATE
    torch_npu
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
)

target_link_directories(sinkhorn_normalize_ops PRIVATE
    ${ACL_LIB_DIR}
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(sinkhorn_normalize_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

target_compile_definitions(sinkhorn_normalize_ops PRIVATE
    SN_TILING_MODE=${SN_TILING_MODE}
    SN_CORE_QUERY=${SN_CORE_QUERY}
)
message(STATUS "SN_TILING_MODE=${SN_TILING_MODE}  SN_CORE_QUERY=${SN_CORE_QUERY}")

message(STATUS "TORCH_LIBRARY module 'libsinkhorn_normalize_ops.so' will be built")
