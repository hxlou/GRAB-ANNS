#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include "cagra.cuh"
#include "config.cuh"
#include "compute_distance.cuh" // 引入距离计算库
#include "bitonic.cuh"          // 引入双调排序库

#include <thrust/sort.h>        // Thrust 排序
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

namespace cagra {

namespace detail {


__global__ void insert_refine_kernel_warp(const float* d_queries,       // [batch, dim]
                                          const float* d_dataset,       // [N, dim]
                                          const int64_t* d_indices,     // [batch, search_k]
                                          float* d_dists,               // [batch, search_k]
                                          size_t num_dataset,
                                          int dim,                      // 固定 1024
                                          int search_k,
                                          int total_pairs);              // 总任务数

template <int N>
__global__ void insert_sort_kernel(int64_t* d_indices, 
                                   float* d_dists, 
                                   int num_queries, 
                                   int search_k);
} // namespace detail

void find_near_nodes(const float* d_dataset,       
                            size_t num_existing,          
                            size_t num_new,               
                            const float* d_new_data,      
                            const uint32_t* d_graph,      
                            uint32_t graph_degree,        
                            uint32_t search_k,            
                            SearchParams params,
                            int64_t* d_out_indices,       
                            float* d_out_dists);

void update_topology_random_cpu(uint32_t* h_graph,
                                const int64_t* h_search_indices,
                                size_t num_existing,
                                size_t num_new,
                                uint32_t graph_degree,
                                uint32_t search_k);

// =============================================================================
// Kernel A: 填充新节点的出边 (Part A)
// 每个线程处理一个新节点的一条出边
// =============================================================================
__global__ void fill_new_nodes_kernel(uint32_t* d_graph,
                                      const int64_t* d_search_indices,
                                      size_t num_existing,
                                      size_t num_new,
                                      uint32_t graph_degree,
                                      uint32_t search_k);
                                      
// =============================================================================
// Kernel B1: 生成反向更新请求列表
// 遍历 search_indices，生成 (Target_Old_ID, Source_New_ID) 对
// =============================================================================
__global__ void generate_update_requests_kernel(const int64_t* d_search_indices,
                                                uint32_t* request_keys,   // Target (Old Node)
                                                uint32_t* request_vals,   // Source (New Node)
                                                size_t num_existing,
                                                size_t num_new,
                                                uint32_t search_k);

// =============================================================================
// Kernel B2: 应用更新 (One Thread per Old Node)
// 修改版：支持概率性更新 Protected 区域
// =============================================================================
__global__ void apply_topology_updates_kernel(uint32_t* d_graph,
                                              const uint32_t* sorted_keys, // Target
                                              const uint32_t* sorted_vals, // Source
                                              uint32_t total_requests,
                                              uint32_t graph_degree);

// =============================================================================
// Host 接口实现
// =============================================================================
void update_topology_gpu_v1(uint32_t* d_graph,
                                const int64_t* d_search_indices,
                                size_t num_existing,
                                size_t num_new,
                                uint32_t graph_degree,
                                uint32_t search_k);

} // namespace cagra