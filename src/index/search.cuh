#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "cagra.cuh"
#include "config.cuh"
#include "hashmap.cuh"
#include "bitonic.cuh"
#include "compute_distance.cuh"

namespace cagra {
namespace device {

__device__ unsigned long long get_global_time();

__device__ __forceinline__ void pickup_next_parents(uint32_t* terminate_flag,
                                                    uint32_t* next_parent_indices,
                                                    uint32_t* internal_topk_indices,
                                                    uint32_t internal_topk_size,
                                                    uint32_t search_width);

template <int N> // N = Capacity / 32
__device__ __forceinline__ void load_sort_store(float* smem_dists, uint32_t* smem_indices, uint32_t capacity);

__global__ void search_kernel(
    uint32_t* result_indices_ptr,       // [num_queries, topk] 输出索引
    float* result_distances_ptr,        // [num_queries, topk] 输出距离
    const float* queries_ptr,           // [num_queries, dim] 查询向量
    const float* dataset_ptr,           // [N, dim] 数据集
    const uint32_t* knn_graph,          // [N, degree] CAGRA 图
    const uint32_t* seed_ptr,           // Seed (可选)
    const uint32_t num_seeds_per_query,           // 单个 query 的 Seeds 数量
    uint32_t* num_executed_iterations,  // 调试用
    
    // --- 运行时参数 ---
    uint32_t num_queries,
    size_t num_dataset,
    uint32_t dim,               
    uint32_t graph_degree,      
    uint32_t topk,              
    uint32_t itopk_size,        
    uint32_t search_width,      
    uint32_t max_iterations,    
    uint32_t num_seeds,         
    uint64_t rand_xor_mask,     
    uint32_t hash_bitlen,
    uint32_t* pre_hashmap,
    uint32_t queue_capacity     // 必须是 32 的倍数 (通常也是 2 的幂次)
);

__global__ void search_kernel_bucket(
    uint32_t* result_indices_ptr,       
    float* result_distances_ptr,        
    const float* queries_ptr,           
    const float* dataset_ptr,           
    const uint32_t* knn_graph,          
    const uint32_t* seed_ptr,           
    uint32_t num_provided_seeds,        
    uint32_t* num_executed_iterations,  
    
    // --- 核心参数 ---
    uint32_t num_queries,
    size_t num_dataset,
    uint32_t dim,               
    uint32_t graph_stride,      // 图的物理宽度 (32)
    uint32_t active_degree,     // 实际使用的逻辑宽度 (28 - Local Edge)
    
    uint32_t topk,              
    uint32_t itopk_size,        
    uint32_t search_width,      
    uint32_t max_iterations,    
    uint32_t num_seeds,         
    uint64_t rand_xor_mask,     
    uint32_t hash_bitlen,
    uint32_t* pre_hashmap,   
    uint32_t queue_capacity     
);

} // namespace device
} // namespace cagra