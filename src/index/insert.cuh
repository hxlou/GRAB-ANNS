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

namespace cagra {

namespace detail {

// =============================================================================
// Helper Kernel 1: 计算 Query 与 候选邻居 的精确 L2 距离 (Warp 级并行版)
// =============================================================================
// 一个 Warp 处理一个 (Query, Neighbor) 对
__global__ void insert_refine_kernel_warp(const float* d_queries,       // [batch, dim]
                                          const float* d_dataset,       // [N, dim]
                                          const int64_t* d_indices,     // [batch, search_k]
                                          float* d_dists,               // [batch, search_k]
                                          size_t num_dataset,
                                          int dim,                      // 固定 1024
                                          int search_k,
                                          int total_pairs)              // 总任务数
{
    // 计算当前 Warp 的全局 ID
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= total_pairs) return;

    // 1. 获取任务信息 (Warp 内所有线程通过广播读取同一个索引)
    int64_t neighbor_id = d_indices[warp_id];

    // 2. 准备写入结果
    float dist = 3.40282e38f; // MAX_FLOAT

    // 3. 如果索引有效，调用 Warp 级距离计算函数
    if (neighbor_id >= 0 && neighbor_id < num_dataset) {
        // 计算 Query 索引 (warp_id 对应扁平化的 pair index)
        int query_idx = warp_id / search_k;
        
        const float* vec_q = d_queries + (size_t)query_idx * dim;
        const float* vec_n = d_dataset + (size_t)neighbor_id * dim;

        // 调用 device::calc_l2_dist_1024 (Warp Collective)
        dist = cagra::device::calc_l2_dist_1024(vec_q, vec_n);
    }

    // 4. 写回结果 (只有 Lane 0 负责写入，避免竞争)
    if (lane_id == 0) {
        d_dists[warp_id] = dist;
    }
}

// =============================================================================
// Helper Kernel 2: 对搜索结果进行 GPU 排序 (Bitonic Sort)
// =============================================================================
// 一个 Warp 处理一个 Query 的所有结果 (search_k 个)
// N = Capacity / 32 (每个线程负责的元素数量)
template <int N>
__global__ void insert_sort_kernel(int64_t* d_indices, 
                                   float* d_dists, 
                                   int num_queries, 
                                   int search_k) 
{
    // 1. 确定当前 Warp 负责哪个 Query
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_queries) return;

    // 2. 定义寄存器数组
    float key[N];
    uint32_t val[N]; // 临时使用 uint32 进行排序

    // 指针偏移到当前 Query 的起始位置
    int64_t* my_indices_ptr = d_indices + (size_t)warp_id * search_k;
    float* my_dists_ptr = d_dists + (size_t)warp_id * search_k;

    // 3. 加载数据 (Global Memory -> Registers)
    // Striped loading: Thread 0 reads 0, 32, 64...
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id + i * 32; 

        if (idx < search_k) {
            key[i] = my_dists_ptr[idx];
            // 强转 int64 -> uint32 (CAGRA 规模下 ID < 40亿，安全)
            // 如果原来的 ID 是 -1 (无效)，转为 uint32 会变成 MAX_UINT，排序时会自动沉底(如果按dist排)
            // 但这里我们主要按 key (dist) 排序，dist 无效时已经是 MAX_FLOAT
            val[i] = static_cast<uint32_t>(my_indices_ptr[idx]);
        } else {
            // Padding 部分填充最大值，使其沉底
            key[i] = 3.40282e38f; // MAX_FLOAT
            val[i] = 0xFFFFFFFF;
        }
    }

    // 4. 执行双调排序 (升序)
    cagra::bitonic::warp_sort<float, uint32_t, N>(key, val, true);

    // 5. 写回数据 (Registers -> Global Memory)
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id + i * 32;
        
        // 只写回有效部分，Padding 丢弃
        if (idx < search_k) {
            my_dists_ptr[idx] = key[i];
            my_indices_ptr[idx] = static_cast<int64_t>(val[i]);
        }
    }
}

} // namespace detail

// =============================================================================
// Helper: 内部搜索函数 (用于 Insert 阶段寻找最近邻)
// =============================================================================
inline void find_near_nodes(const float* d_dataset,       
                            size_t num_existing,          
                            size_t num_new,               
                            const float* d_new_data,      
                            const uint32_t* d_graph,      
                            uint32_t graph_degree,        
                            uint32_t search_k,            
                            SearchParams params,
                            int64_t* d_out_indices,       
                            float* d_out_dists)           
{
    // 增大 Batch Size，全 GPU 流程可以处理更大的批次
    const size_t batch_size = 64; 

    for (size_t offset = 0; offset < num_new; offset += batch_size) {
        size_t current_batch = std::min(batch_size, num_new - offset);

        const float* curr_queries_ptr = d_new_data + offset * cagra::config::DIM;
        int64_t* curr_indices_ptr = d_out_indices + offset * search_k;
        float* curr_dists_ptr = d_out_dists + offset * search_k;

        // 1. 初步搜索 (GPU Search)
        cagra::search(
            d_dataset,
            num_existing,
            d_graph,
            graph_degree,
            curr_queries_ptr,
            (int64_t)current_batch,
            (int64_t)search_k,
            params,
            curr_indices_ptr,
            curr_dists_ptr
        );

        // 2. 精确距离计算 (GPU Calc)
        // 这里的配置是：1 个 Warp 处理 1 对 (Query, Candidate)
        int total_pairs = current_batch * search_k;
        int threads_calc = total_pairs * 32; 
        int block_calc = 256; 
        int grid_calc = (threads_calc + block_calc - 1) / block_calc;

        detail::insert_refine_kernel_warp<<<grid_calc, block_calc>>>(
            curr_queries_ptr,
            d_dataset,
            curr_indices_ptr,
            curr_dists_ptr,
            num_existing,
            cagra::config::DIM, // 1024
            search_k,
            total_pairs
        );
        CUDA_CHECK(cudaGetLastError());

        // 3. 排序 (GPU Sort)
        // 这里的配置是：1 个 Warp 处理 1 个 Query
        int threads_sort = current_batch * 32;
        int block_sort = 256;
        int grid_sort = (threads_sort + block_sort - 1) / block_sort;

        // 根据 search_k 动态分发到对应的模板 Kernel
        // N 必须满足 32 * N >= search_k，且为 2 的幂次
        if (search_k <= 64) {
            detail::insert_sort_kernel<2><<<grid_sort, block_sort>>>(curr_indices_ptr, curr_dists_ptr, current_batch, search_k);
        } else if (search_k <= 128) {
            detail::insert_sort_kernel<4><<<grid_sort, block_sort>>>(curr_indices_ptr, curr_dists_ptr, current_batch, search_k);
        } else if (search_k <= 256) {
            detail::insert_sort_kernel<8><<<grid_sort, block_sort>>>(curr_indices_ptr, curr_dists_ptr, current_batch, search_k);
        } else {
            // Fallback for larger k, e.g., 512
            detail::insert_sort_kernel<16><<<grid_sort, block_sort>>>(curr_indices_ptr, curr_dists_ptr, current_batch, search_k);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 等待所有 Batch 完成
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cagra