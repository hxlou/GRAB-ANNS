#include "insert.cuh"
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
void find_near_nodes(const float* d_dataset,       
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

// =============================================================================
// Helper: CPU 侧拓扑更新 (随机替换策略) 弃用
// =============================================================================
void update_topology_random_cpu(uint32_t* h_graph,
                                const int64_t* h_search_indices,
                                size_t num_existing,
                                size_t num_new,
                                uint32_t graph_degree,
                                uint32_t search_k)
{
    // 保护区大小：前一半不动，后一半可以随机替换
    const uint32_t num_protected = graph_degree / 2;
    const uint32_t num_unprotected = graph_degree - num_protected;

    // OpenMP 并行处理每个新插入的节点
    #pragma omp parallel for
    for (size_t i = 0; i < num_new; ++i) {
        // 新节点的逻辑 ID 是接着旧节点后面的
        size_t new_node_id = num_existing + i;
        
        // 伪随机数生成
        uint64_t rng_state = new_node_id * 0x9e3779b97f4a7c15; 

        // --- Part A: 填充新节点的出边 (我指向谁) ---
        uint32_t* new_node_neighbors = h_graph + new_node_id * graph_degree;
        
        // 初始化
        for(uint32_t k=0; k<graph_degree; ++k) new_node_neighbors[k] = 0xFFFFFFFF;

        // 直接从搜索结果中取 Top-Degree
        for (uint32_t k = 0; k < graph_degree; ++k) {
            int64_t idx = h_search_indices[i * search_k + k];
            // 确保指向的是有效的旧节点
            if (idx >= 0 && idx < (int64_t)num_existing) {
                new_node_neighbors[k] = (uint32_t)idx;
            }
        }

        // --- Part B: 更新反向边 (随机替换谁指向我) ---
        for (int m = 0; m < search_k; ++m) {
            int64_t neighbor_idx_64 = h_search_indices[i * search_k + m];
            if (neighbor_idx_64 < 0 || neighbor_idx_64 >= (int64_t)num_existing) continue;
            
            uint32_t old_id = (uint32_t)neighbor_idx_64;
            uint32_t* old_neighbors = h_graph + old_id * graph_degree;

            // 1. 查重
            bool exists = false;
            for (uint32_t x = 0; x < graph_degree; ++x) {
                if (old_neighbors[x] == (uint32_t)new_node_id) {
                    exists = true; 
                    break;
                }
            }

            if (!exists) {
                // 2. 随机替换逻辑 (仅在 Unprotected 区域)
                rng_state = rng_state * 6364136223846793005ULL + 1;
                uint32_t random_offset = (rng_state >> 32) % num_unprotected;
                uint32_t replace_idx = num_protected + random_offset;

                old_neighbors[replace_idx] = (uint32_t)new_node_id;
            }
        }
    }
}

// =============================================================================
// Kernel A: 填充新节点的出边 (Part A)
// 每个线程处理一个新节点的一条出边
// =============================================================================
__global__ void fill_new_nodes_kernel(uint32_t* d_graph,
                                      const int64_t* d_search_indices,
                                      size_t num_existing,
                                      size_t num_new,
                                      uint32_t graph_degree,
                                      uint32_t search_k)
{
    // 二维 Grid: x 轴对应 graph_degree，y 轴对应 num_new
    uint32_t k = threadIdx.x + blockIdx.x * blockDim.x; // 第几个邻居槽位
    uint32_t i = blockIdx.y;                            // 第几个新节点

    if (i >= num_new || k >= graph_degree) return;

    // 新节点的全局 ID
    size_t new_node_id = num_existing + i;
    
    // 初始化为无效
    uint32_t neighbor_val = 0xFFFFFFFF;

    // 如果 k 在搜索结果范围内，则尝试获取
    if (k < search_k) {
        // search_indices 布局: [num_new, search_k]
        int64_t idx = d_search_indices[i * search_k + k];
        
        // 确保指向的是有效的旧节点 (不能指向自己，也不能指向其他新节点——根据原逻辑)
        if (idx >= 0 && idx < (int64_t)num_existing) {
            neighbor_val = (uint32_t)idx;
        }
    }

    // 写入图
    d_graph[new_node_id * graph_degree + k] = neighbor_val;
}

// =============================================================================
// Kernel B1: 生成反向更新请求列表
// 遍历 search_indices，生成 (Target_Old_ID, Source_New_ID) 对
// =============================================================================
__global__ void generate_update_requests_kernel(const int64_t* d_search_indices,
                                                uint32_t* request_keys,   // Target (Old Node)
                                                uint32_t* request_vals,   // Source (New Node)
                                                size_t num_existing,
                                                size_t num_new,
                                                uint32_t search_k)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_requests = num_new * search_k;

    if (tid >= total_requests) return;

    // 计算由哪个新节点产生的请求
    uint32_t new_node_idx = tid / search_k; // 第几个新节点 (0 ~ num_new-1)
    
    // 获取搜索结果中的邻居 ID
    int64_t neighbor_idx_64 = d_search_indices[tid];

    // 只有当邻居是有效的旧节点时，才生成请求
    if (neighbor_idx_64 >= 0 && neighbor_idx_64 < (int64_t)num_existing) {
        request_keys[tid] = (uint32_t)neighbor_idx_64;      // Target
        request_vals[tid] = (uint32_t)(num_existing + new_node_idx); // Source (Global ID)
    } else {
        // 无效请求，标记 Key 为最大值，排序后会被推到最后
        request_keys[tid] = 0xFFFFFFFF;
        request_vals[tid] = 0xFFFFFFFF;
    }
}

// =============================================================================
// Kernel B2: 应用更新 (One Thread per Old Node)
// 修改版：支持概率性更新 Protected 区域
// =============================================================================
__global__ void apply_topology_updates_kernel(uint32_t* d_graph,
                                              const uint32_t* sorted_keys, // Target
                                              const uint32_t* sorted_vals, // Source
                                              uint32_t total_requests,
                                              uint32_t graph_degree)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= total_requests) return;

    uint32_t target_id = sorted_keys[tid];

    // 如果是无效 Key，直接跳过
    if (target_id == 0xFFFFFFFF) return;

    // 边界检测：我是不是当前这段相同 Key 的第一个线程？
    bool is_leader = (tid == 0) || (sorted_keys[tid - 1] != target_id);

    if (is_leader) {
        // 1. 定位要修改的旧节点邻居列表
        uint32_t* neighbors = d_graph + target_id * graph_degree;
        
        // 2. 准备参数
        const uint32_t num_protected = graph_degree / 2;
        const uint32_t num_unprotected = graph_degree - num_protected;
        
        // 随机数状态初始化
        uint64_t rng_state = target_id * 0x9e3779b97f4a7c15 + tid;

        // 3. 遍历所有指向该 Target 的新节点
        for (uint32_t idx = tid; idx < total_requests; ++idx) {
            // 如果遇到不同的 Key，说明当前 Target 的请求处理完了
            if (sorted_keys[idx] != target_id) break;

            uint32_t new_node_global_id = sorted_vals[idx];

            // 3.1 查重
            bool exists = false;
            for (uint32_t x = 0; x < graph_degree; ++x) {
                if (neighbors[x] == new_node_global_id) {
                    exists = true; 
                    break;
                }
            }

            // 3.2 替换逻辑 (修改部分)
            if (!exists) {
                // --- Step A: 决定是否更新 Protected 区域 (10% 概率) ---
                rng_state = rng_state * 6364136223846793005ULL + 1; // 第一次步进
                uint32_t r1 = rng_state >> 32;
                
                // 取模 10，如果等于 0 则为 10% 的概率
                bool update_protected = (r1 % 10 == 0); 

                // --- Step B: 决定具体的替换位置 ---
                rng_state = rng_state * 6364136223846793005ULL + 1; // 第二次步进，保证位置随机性独立
                uint32_t r2 = rng_state >> 32;
                
                uint32_t replace_idx;

                if (update_protected) {
                    // 10% 概率：随机替换前半部分 [0, num_protected)
                    replace_idx = r2 % num_protected;
                } else {
                    // 90% 概率：随机替换后半部分 [num_protected, graph_degree)
                    replace_idx = num_protected + (r2 % num_unprotected);
                }

                neighbors[replace_idx] = new_node_global_id;
            }
        }
    }
}
// =============================================================================
// Host 接口实现
// =============================================================================
void update_topology_gpu_v1(uint32_t* d_graph,
                                const int64_t* d_search_indices,
                                size_t num_existing,
                                size_t num_new,
                                uint32_t graph_degree,
                                uint32_t search_k)
{
    // -------------------------------------------------------
    // Part A: 填充新节点的出边 (Fill Outgoing Edges)
    // -------------------------------------------------------
    {
        dim3 block(32, 1); // x轴覆盖 degree (假设 degree <= 32 或分块)
        // 如果 degree > 32，可以让 x 轴循环处理。这里简单起见假设 degree 32/64
        if (graph_degree > 32) block.x = 64;
        
        dim3 grid(1, (uint32_t)num_new);
        
        fill_new_nodes_kernel<<<grid, block>>>(
            d_graph, d_search_indices, num_existing, num_new, graph_degree, search_k
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // -------------------------------------------------------
    // Part B: 更新反向边 (Update Reverse/Incoming Edges)
    // 策略: Generate -> Sort -> Apply
    // -------------------------------------------------------
    
    // 1. 申请临时显存存放请求列表
    uint32_t total_requests = num_new * search_k;
    uint32_t* d_req_keys = nullptr; // Target (Old ID)
    uint32_t* d_req_vals = nullptr; // Source (New ID)
    
    CUDA_CHECK(cudaMalloc(&d_req_keys, total_requests * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_req_vals, total_requests * sizeof(uint32_t)));

    // 2. 生成请求
    {
        uint32_t block_size = 256;
        uint32_t grid_size = (total_requests + block_size - 1) / block_size;
        
        generate_update_requests_kernel<<<grid_size, block_size>>>(
            d_search_indices, d_req_keys, d_req_vals, 
            num_existing, num_new, search_k
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. 排序 (按 Target Old ID 排序)
    // 使用 Thrust 进行 Radix Sort
    thrust::device_ptr<uint32_t> t_keys(d_req_keys);
    thrust::device_ptr<uint32_t> t_vals(d_req_vals);
    thrust::sort_by_key(t_keys, t_keys + total_requests, t_vals);

    // 4. 应用更新 (Apply)
    {
        uint32_t block_size = 256;
        uint32_t grid_size = (total_requests + block_size - 1) / block_size;

        apply_topology_updates_kernel<<<grid_size, block_size>>>(
            d_graph, d_req_keys, d_req_vals, total_requests, graph_degree
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 5. 清理临时显存
    CUDA_CHECK(cudaFree(d_req_keys));
    CUDA_CHECK(cudaFree(d_req_vals));
}

} // namespace cagra