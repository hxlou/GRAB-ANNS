cmake_minimum_required(VERSION 3.20)

# 设置 GPU 架构 (native 表示针对当前机器的显卡自动优化)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(LightCagra LANGUAGES CXX CUDA)

# ==========================================
# 1. 编译选项设置
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ==========================================
# 2. 寻找依赖 (FAISS & CUDA)
# ==========================================
# 告诉 CMake 去 CONDA 环境里找包
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
endif()

# 查找 CUDA 工具包 (用于 cublas, cudart)
find_package(CUDAToolkit REQUIRED)

# 查找 FAISS
# Conda 环境下通常没有标准的 Findfaiss.cmake，我们直接查找库文件
find_library(FAISS_LIB faiss REQUIRED)

# ==========================================
# 3. 定义核心库 (light_cagra)
# ==========================================
add_library(light_cagra
    src/index/cagra.cu
)

# 包含头文件路径
target_include_directories(light_cagra PUBLIC 
    src
    src/index
)

# 链接依赖
target_link_libraries(light_cagra PUBLIC
    ${FAISS_LIB}      # 链接 FAISS
    CUDA::cudart      # FAISS 依赖
    CUDA::cublas      # FAISS 依赖
)

# ==========================================
# 4. 定义测试可执行文件
# ==========================================
add_executable(test_knn_build tests/test_knn_build.cpp)
target_link_libraries(test_knn_build PRIVATE light_cagra)

add_executable(test_cagra_build tests/test_cagra_build.cpp)
target_link_libraries(test_cagra_build PRIVATE light_cagra)

add_executable(test_bitonic tests/test_bitonic.cu)
target_link_libraries(test_bitonic PRIVATE light_cagra)

add_executable(test_hashmap tests/test_hashmap.cu)
target_link_libraries(test_hashmap PRIVATE light_cagra)

add_executable(test_cagra tests/test_cagra.cpp)
target_link_libraries(test_cagra PRIVATE light_cagra)

add_executable(test_sort_impl tests/test_sort_impl.cu)
target_link_libraries(test_sort_impl PRIVATE light_cagra)