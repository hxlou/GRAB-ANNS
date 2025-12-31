cmake_minimum_required(VERSION 3.20)

# =========================================================
# 【关键修改】在 project() 之前强制指定 Conda 编译器
# =========================================================
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_ROOT "$ENV{CONDA_PREFIX}")
    message(STATUS ">> Detecting Conda Env: ${CONDA_ROOT}")
    
    # Conda 的编译器通常位于 $CONDA_PREFIX/bin 下，且有特定的前缀
    set(CONDA_C_COMPILER   "${CONDA_ROOT}/bin/x86_64-conda-linux-gnu-gcc")
    set(CONDA_CXX_COMPILER "${CONDA_ROOT}/bin/x86_64-conda-linux-gnu-g++")
    
    if(EXISTS "${CONDA_C_COMPILER}" AND EXISTS "${CONDA_CXX_COMPILER}")
        message(STATUS ">> Forcing use of Conda Compilers to avoid GLIBC conflict:")
        message(STATUS "   > C Compiler:   ${CONDA_C_COMPILER}")
        message(STATUS "   > CXX Compiler: ${CONDA_CXX_COMPILER}")
        
        # 强制设置 C 和 C++ 编译器
        set(CMAKE_C_COMPILER "${CONDA_C_COMPILER}")
        set(CMAKE_CXX_COMPILER "${CONDA_CXX_COMPILER}")
        
        # 【非常重要】告诉 NVCC 使用 Conda 的 g++ 作为宿主编译器
        set(CMAKE_CUDA_HOST_COMPILER "${CONDA_CXX_COMPILER}")
    else()
        message(WARNING ">> gxx_linux-64 not found in Conda bin! using system compiler (Risk of GLIBC error).")
    endif()
else()
    message(WARNING ">> Not in a Conda environment.")
endif()
# =======

set(CMAKE_CXX_STANDARD 17) # RAFT 需要 C++17 或更高
set(CMAKE_CUDA_STANDARD 17)

# 设置 GPU 架构 (native 表示针对当前机器的显卡自动优化)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(LightCagra LANGUAGES CXX CUDA)

# 设置 CUDA 运行时库为共享库
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 定义 CUDA 设备 ID（可以通过命令行覆盖）
set(CUDA_DEVICE_ID "0" CACHE STRING "CUDA device ID to use")
add_compile_definitions(CUDA_DEVICE_ID=${CUDA_DEVICE_ID})

# # 1. 首先设置构建类型
# set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

# # 2. 设置 CUDA 调试选项（覆盖默认值）
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -lineinfo" CACHE STRING "CUDA debug flags" FORCE)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0 -lineinfo")

# # 3. 设置 C++ 调试选项
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "CXX debug flags" FORCE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

# # 4. 移除任何优化标志
# string(REPLACE "-O2" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(REPLACE "-O3" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(REPLACE "-O2" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# string(REPLACE "-O3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# # 5. 打印确认信息
# message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
# message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
# message(STATUS "CUDA Debug Flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
# message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
# message(STATUS "CXX Debug Flags: ${CMAKE_CXX_FLAGS_DEBUG}")

# ==========================================
# 1. 编译选项设置
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ==========================================
# 2. 依赖查找 (Conda, OpenMP, CUDA, FAISS)
# ==========================================

# 2.1 Conda 环境配置
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_ROOT "$ENV{CONDA_PREFIX}")
    message(STATUS ">> Using Conda Env: ${CONDA_ROOT}")
    
    # 将 Conda 路径设为最高优先级
    list(PREPEND CMAKE_PREFIX_PATH "${CONDA_ROOT}")
    include_directories("${CONDA_ROOT}/include")
    link_directories("${CONDA_ROOT}/lib")
else()
    message(WARNING "Not in a Conda environment. Relying on system paths.")
endif()

# 2.2 查找 OpenMP
find_package(OpenMP REQUIRED)

# 查找 RMM (Conda 安装后会有 rmm-config.cmake)
find_package(rmm REQUIRED)

# 查找 RAFT (Conda 安装后会有 raft-config.cmake)
find_package(raft REQUIRED)

# 2.3 查找 CUDA Toolkit (核心)
# 这提供了 CUDA::cudart (Runtime) 和 CUDA::cuda_driver (Driver API for VMM)
find_package(CUDAToolkit REQUIRED)
# 2. 如果标准目标 CUDA::cuda_driver 不存在，手动创建一个别名
if(NOT TARGET CUDA::cuda_driver)
    message(STATUS ">> CUDA::cuda_driver target missing, trying to find libcuda.so manually...")
    
    # 在系统常用路径查找 libcuda
    find_library(CUDA_DRIVER_LIB 
        NAMES cuda libcuda 
        HINTS 
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib64"
        "/usr/local/cuda/lib64/stubs" # 很多时候在这里
    )

    if(CUDA_DRIVER_LIB)
        message(STATUS ">> Found CUDA Driver manually: ${CUDA_DRIVER_LIB}")
        add_library(CUDA::cuda_driver UNKNOWN IMPORTED)
        set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION "${CUDA_DRIVER_LIB}")
    else()
        message(FATAL_ERROR "Could not find libcuda.so (CUDA Driver API). Please check your driver installation.")
    endif()
endif()
message(STATUS ">> CUDA Toolkit Found: ${CUDAToolkit_FOUND}")
message(STATUS ">> CUDA Driver Lib: ${CUDA_cuda_driver_LIBRARY}")
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 2.4 查找 FAISS
# 优先在 Conda lib 下查找
find_library(FAISS_LIB faiss REQUIRED HINTS "${CONDA_ROOT}/lib")

# ==========================================
# 3. 定义核心库 (light_cagra)
# ==========================================
add_library(light_cagra SHARED
    src/index/cagra.cu
    src/index/cagra_opt.cu
    src/index/insert.cu
    src/index/search.cu
    
    
    src/cagraIndex.cu  # 确保你的文件是 .cu 后缀
    # src/timeStampIndex.cu
    src/cagraIndexOpt.cu
    
)

target_include_directories(light_cagra PUBLIC 
    src 
    src/index
)

# 链接依赖
target_link_libraries(light_cagra PUBLIC
    ${FAISS_LIB}
    CUDA::cudart       # CUDA Runtime API (cudaMalloc, etc.)
    CUDA::cublas       # FAISS 依赖
    CUDA::cuda_driver  # 【重要】CUDA Driver API (cuInit, cuMemCreate for VMM)
    OpenMP::OpenMP_CXX
    raft::raft
    rmm::rmm
)

# ==========================================
# 4. 定义测试辅助函数
# ==========================================
function(add_cagra_test test_name source_file)
    # 根据源文件后缀决定是 C++ 还是 CUDA 链接
    add_executable(${test_name} ${source_file})
    
    target_link_libraries(${test_name} PRIVATE 
        light_cagra 
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
endfunction()

# ==========================================
# 5. 添加测试目标
# ==========================================

# 基础组件测试
add_cagra_test(test_knn_build    tests/test_knn_build.cu)
add_cagra_test(test_bitonic      tests/test_bitonic.cu)
add_cagra_test(test_hashmap      tests/test_hashmap.cu)
add_cagra_test(test_cagra        tests/test_cagra.cu)
# 索引功能测试

add_cagra_test(test_cagra_index  tests/test_cagra_index.cu)
add_cagra_test(test_cagra_insert tests/test_cagra_insert.cu)

# add_cagra_test(test_timestamp_index tests/test_timestamp_index.cu)
# add_cagra_test(test_timestamp_index_random tests/test_timestamp_index_random.cu)

add_cagra_test(test_cagra_opt tests/test_cagra_opt.cu)
add_cagra_test(test_cagra_opt_search_bucket tests/test_cagra_opt_search_bucket.cu)
add_cagra_test(test_cagra_opt_search_range tests/test_cagra_opt_search_range.cu)
add_cagra_test(test_cagra_opt_insert tests/test_cagra_opt_insert.cu)
add_cagra_test(test_cagra_opt_build_auto tests/test_cagra_opt_build_auto.cu)
add_cagra_test(test_cagra_opt_insert_auto tests/test_cagra_opt_insert_auto.cu)

# add_cagra_test(test_faiss_cagra tests/test_faiss_cagra.cu)

# FAISS HNSW 对比测试 (如果有的话)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_faiss_hnsw_insert.cpp")
    add_executable(test_faiss_hnsw tests/test_faiss_hnsw_insert.cpp)
    target_link_libraries(test_faiss_hnsw PRIVATE 
        light_cagra 
        OpenMP::OpenMP_CXX
        CUDA::cudart
    )
endif()