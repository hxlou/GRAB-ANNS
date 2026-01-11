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

        if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(vec_q, vec_n);
        else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(vec_q, vec_n);
        else if (dim == 960) dist = cagra::device::calc_l2_dist_960(vec_q, vec_n);
        else if (dim == 256) dist = cagra::device::calc_l2_dist_256(vec_q, vec_n);
        else if (dim == 128) dist = cagra::device::calc_l2_dist_128(vec_q, vec_n);
        else if (dim == 96) dist = cagra::device::calc_l2_dist_96(vec_q, vec_n);
        else {
            // 对于非特殊维度，调用通用版本
            printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
        }
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
                            uint32_t dim,   
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

        const float* curr_queries_ptr = d_new_data + offset * dim;
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
            dim, // 1024
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



// =============================================================================
//  opt insert 函数实现
// ============================================================================

__global__ void fill_new_nodes_kernel_opt(
    uint32_t* d_graph,              // [Total_N, 32] (写入目标)
    const uint64_t* d_ts,           // [Total_N] 时间戳 (用于验证)
    int64_t* d_local_knn,     // [num_new, search_k_local]
    int64_t* d_global_knn,    // [num_new, search_k_global]
    size_t num_existing,            // 老数据量 (即新数据的起始全局ID)
    size_t num_new,                 // 新数据量
    uint32_t total_degree,          // 32
    uint32_t local_degree,          // 28
    uint32_t search_k_local,        // 搜索结果的宽度 (e.g. 64)
    uint32_t search_k_global        // 搜索结果的宽度 (e.g. 64)
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_new) return;

    // 当前新节点的全局 ID
    size_t my_global_id = num_existing + tid;
    uint64_t my_ts = d_ts[my_global_id];

    // 指向 d_graph 中当前新节点的行
    uint32_t* my_graph_row = d_graph + my_global_id * total_degree;

    // -------------------------------------------------------------
    // Phase 1: 填充 Local Edges [0, local_degree-1]
    // -------------------------------------------------------------
    int filled_local = 0;
    int64_t* my_local_candidates = d_local_knn + tid * search_k_local;

    for (uint32_t k = 0; k < search_k_local; ++k) {
        if (filled_local >= local_degree) break;

        int64_t cand_id_64 = my_local_candidates[k];
        uint32_t cand_id = (uint32_t)cand_id_64;
        int64_t cand_ts = d_ts[cand_id];

        // 过滤无效值
        if (cand_id_64 < 0) {
            printf("invalid local candidate id %ld for new node %d at local rank %u\n", cand_id_64, my_global_id, k);
            continue;
        }
        if (cand_id == my_global_id) {
            // printf("self-loop detected for new node %d at local rank %u\n", my_global_id, k);
            continue;
        } // 不连自己
        if (cand_id >= 0x7fffffff) {
            printf("invalid local candidate id %u for new node %d at local rank %u\n", cand_id, my_global_id, k);
            continue;
        } // 过滤异常大 ID

        if (cand_ts != my_ts) {
            printf("1145114 local candidate %u for new node %lu at local rank %u is from different bucket (ts %ld vs %lu)\n", cand_id, my_global_id, k, cand_ts, my_ts);
            printf("num existing: %lu, num_new: %lu\n", num_existing, num_new);
            continue;
        }

        // 强制时间戳约束：必须同桶
        if (cand_ts == my_ts) {
            my_graph_row[filled_local++] = (uint32_t)cand_id;
            my_local_candidates[k] = -1; // 标记为已使用
        }
    }

    // 补齐 Local
    for (int i = 0; i < search_k_local && filled_local < local_degree; ++i) {
        int64_t cand_id_64 = my_local_candidates[i];
        uint32_t cand_id = (uint32_t)cand_id_64;

        if (cand_id_64 < 0) continue; // 已使用或无效
        if (cand_id == my_global_id) continue; // 不连自己

        int64_t cand_ts = d_ts[cand_id];
        if (cand_ts != my_ts) continue; // 必须同桶

        // 补齐
        my_graph_row[filled_local++] = (uint32_t)cand_id;
        my_local_candidates[i] = -1; // 标记为已使用
    }

    // -------------------------------------------------------------
    // Phase 2: 填充 Remote Edges [local_degree, total_degree-1]
    // -------------------------------------------------------------
    int filled_remote = 0;
    int max_remote = total_degree - local_degree;
    int64_t* my_global_candidates = d_global_knn + tid * search_k_global;

    for (uint32_t k = 0; k < search_k_global; ++k) {
        if (filled_remote >= max_remote) break;

        int64_t cand_id_64 = my_global_candidates[k];
        uint32_t cand_id = (uint32_t)cand_id_64;

        if (cand_id_64 < 0) continue;
        if (cand_id == my_global_id) continue;

        // 查重：不要把已经在 Local 里的点再加一遍
        // 虽然一个是 Local Search 来的，一个是 Global Search 来的，但可能有重叠
        bool exists_in_local = false;
        for (int i = 0; i < local_degree; ++i) {
            if (my_graph_row[i] == cand_id) {
                exists_in_local = true;
                break;
            }
        }
        if (exists_in_local) continue;

        // 强制时间戳约束：优先异桶，或者强制异桶？
        // 策略：既然叫 Remote Edge，我们强制要求它是异桶的。
        // 如果实在找不到异桶的，再考虑同桶（这里先实现强制异桶）
        if (d_ts[cand_id] != my_ts) {
            my_graph_row[local_degree + filled_remote] = (uint32_t)cand_id;
            filled_remote++;
            // 标记为已使用
            my_global_candidates[k] = -1;
        }
    }

    // 补齐 Remote
    for (int i = 0; i < search_k_global && filled_remote < max_remote; ++i) {
        int64_t cand_id_64 = my_global_candidates[i];
        uint32_t cand_id = (uint32_t)cand_id_64;

        if (cand_id_64 < 0) continue; // 已使用或无效
        if (cand_id == my_global_id) continue; // 不连自己

        my_graph_row[local_degree + filled_remote] = (uint32_t)cand_id;
        filled_remote++;
        // 标记为已使用
        my_global_candidates[i] = -1;
    }
}

__global__ void generate_update_requests_kernel_opt(
    const uint32_t* d_graph,         // [Total_N, 32] (读取新节点的行)
    
    // --- Local Mailbox ---
    uint32_t* d_local_req_counts,    // [num_existing]
    uint32_t* d_local_req_lists,     // [num_existing, max_requests]
    
    // --- Remote Mailbox ---
    uint32_t* d_remote_req_counts,   // [num_existing]
    uint32_t* d_remote_req_lists,    // [num_existing, max_requests]
    
    size_t num_existing,             // 老节点数量 (Mailbox 的大小)
    size_t num_new,                  // 新节点数量
    size_t target_ts,
    const uint64_t* d_ts,                   // [Total_N] 时间戳 (用于验证)
    uint32_t total_degree,           // 32
    uint32_t local_degree,           // 28
    uint32_t local_max_requests,            // Mailbox 容量
    uint32_t remote_max_requests           // Mailbox 容量
) {
    // 1. 线程索引：每个线程处理一个【新节点】
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_new) return;

    // 当前新节点的全局 ID
    uint32_t src_id = (uint32_t)(num_existing + tid);

    // 指向我在图中的那一行
    const uint32_t* my_neighbors = d_graph + (size_t)src_id * total_degree;

    // 2. 遍历我的所有出边
    for (int i = 0; i < total_degree; ++i) {
        uint32_t dest_id = my_neighbors[i];

        // 2.1 过滤无效边
        if (dest_id == 0xFFFFFFFF) continue;

        // 2.2 确保只向【老节点】发请求
        // 既然 Mailbox 是按 num_existing 分配的，必须加这个判断防止越界
        // if (dest_id >= num_existing) continue;

        // 2.3 判定类型：直接根据索引 i 判断
        // [0, local_degree-1] -> Local
        // [local_degree, total_degree-1] -> Remote
        if (i < local_degree) {
            // --- Local Mailbox ---
            uint32_t pos = atomicAdd(&d_local_req_counts[dest_id], 1);
            if (pos < local_max_requests) {
                // 写入请求：我是 src_id，我想连你
                d_local_req_lists[(size_t)dest_id * local_max_requests + pos] = src_id;
            }
        } else {
            // --- Remote Mailbox ---
            // 如果dest id与我们当前插入的本桶的数据重合，说明完全没插入！此处需要过滤掉
            if (d_ts[dest_id] == target_ts) {
                continue;
            }
            uint32_t pos = atomicAdd(&d_remote_req_counts[dest_id], 1);
            if (pos < remote_max_requests) {
                d_remote_req_lists[(size_t)dest_id * remote_max_requests + pos] = src_id;
            }
        }
    }
}

__device__ __forceinline__ uint32_t wang_hash_insert(uint32_t seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__global__ void apply_topology_updates_kernel_opt(
    uint32_t* d_graph,               // [Total_N, 32]
    
    // --- Local Mailbox ---
    const uint32_t* d_local_req_counts,
    const uint32_t* d_local_req_lists,
    uint32_t max_requests_local,
    
    // --- Remote Mailbox ---
    const uint32_t* d_remote_req_counts,
    const uint32_t* d_remote_req_lists,
    uint32_t max_requests_remote,

    const uint64_t* d_ts,                // [Total_N] 时间戳 (用于验证)
    size_t num_existing,             
    uint32_t total_degree,           // 32
    uint32_t local_degree            // 28
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_existing) return;

    // 1. 准备参数
    uint32_t* my_row = d_graph + tid * total_degree;
    uint32_t rng_state = wang_hash_insert((uint32_t)tid);

    // 定义概率阈值 (10%)
    const uint32_t PROB_THRESHOLD = 10; 

    // -------------------------------------------------------------------------
    // 处理 Local 请求
    // -------------------------------------------------------------------------
    uint32_t local_count = d_local_req_counts[tid];
    if (local_count > 0) {
        if (local_count > max_requests_local) local_count = max_requests_local;
        const uint32_t* my_requests = d_local_req_lists + (size_t)tid * max_requests_local;
        
        // 计算区域边界
        uint32_t local_half = local_degree / 2;
        
        // Strong 区域: [0, local_half)
        // Weak   区域: [local_half, local_degree)
        uint32_t strong_len = local_half;
        uint32_t weak_len = local_degree - local_half;

        for (uint32_t i = 0; i < local_count; ++i) {
            uint32_t candidate = my_requests[i];
            
            // 1. 掷骰子决定攻击哪个区域
            rng_state = wang_hash_insert(rng_state);
            bool attack_strong = (rng_state % 100) < PROB_THRESHOLD;

            // 2. 确定攻击范围
            uint32_t target_start = attack_strong ? 0 : local_half;
            uint32_t target_len   = attack_strong ? strong_len : weak_len;

            // 3. 在目标范围内随机替换
            if (target_len > 0) {
                rng_state = wang_hash_insert(rng_state); // 更新随机数用于选槽位
                int slot = target_start + (rng_state % target_len);
                my_row[slot] = candidate;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 处理 Remote 请求
    // -------------------------------------------------------------------------
    uint32_t remote_count = d_remote_req_counts[tid];
    if (remote_count > 0) {
        if (remote_count > max_requests_remote) remote_count = max_requests_remote;
        const uint32_t* my_requests = d_remote_req_lists + (size_t)tid * max_requests_remote;

        // 计算区域边界
        uint32_t remote_degree = total_degree - local_degree;
        uint32_t remote_half = remote_degree / 2;
        
        // Strong 区域: [local_degree, local_degree + remote_half)
        // Weak   区域: [local_degree + remote_half, total_degree)
        uint32_t strong_start = local_degree;
        uint32_t strong_len = remote_half;
        
        uint32_t weak_start = local_degree + remote_half;
        uint32_t weak_len = remote_degree - remote_half;

        for (uint32_t i = 0; i < remote_count; ++i) {
            uint32_t candidate = my_requests[i];

            // 1. 掷骰子
            // rng_state = wang_hash_insert(rng_state);
            // bool attack_strong = (rng_state % 100) < PROB_THRESHOLD;

            // 2. 确定攻击范围 前 1/3 写入nearby节点，即与我的桶距离只有全量的20%的节点，后2/3写入远程节点
            int64_t my_ts = (int64_t)d_ts[tid];
            int64_t cand_ts = (int64_t)d_ts[candidate];
            bool is_nearby = (cand_ts >= my_ts - 20) && (cand_ts <= my_ts + 20); // 与我的桶距离只有全量的20%的节点
            // uint32_t target_start = attack_strong ? strong_start : weak_start;
            // uint32_t target_len   = attack_strong ? strong_len : weak_len;
            uint32_t nearby_start = local_degree;
            uint32_t nearby_len = remote_degree / 3;
            uint32_t far_start = local_degree + remote_degree / 3;
            uint32_t far_len = remote_degree - nearby_len;
            uint32_t target_start = is_nearby ? nearby_start : far_start;
            uint32_t target_len   = is_nearby ? nearby_len   : far_len;

            // 3. 随机替换
            if (target_len > 0) {
                rng_state = wang_hash_insert(rng_state);
                int slot = target_start + (rng_state % target_len);
                my_row[slot] = candidate;
            }
        }
    }
}


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32
#define MAX_CANDIDATES 128 // 候选列表最大容量 (Local + Buffer)

struct Candidate {
    uint32_t id;
    float dist;
};

// =========================================================================
// 距离计算：SMEM vs Global
// =========================================================================
// vec_a 存储在 Shared Memory (我的向量)
// vec_b 存储在 Global Memory (邻居/候选向量)
// 适用于任意维度 dim
__device__ __forceinline__ float calc_dist_smem_global(
    const float* s_vec_a, 
    const float* g_vec_b, 
    int dim
) {
    int lane = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 循环处理，步长为 32
    for (int i = lane; i < dim; i += 32) {
        float diff = s_vec_a[i] - g_vec_b[i];
        sum_sq += diff * diff;
    }

    // Warp Reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    return sum_sq;
}

// =========================================================================
// 简单的 Warp 内单线程排序 (Lane 0 执行)
// =========================================================================
__device__ void sort_candidates_lane0(Candidate* list, int count) {
    // 简单的插入排序，适合小数组
    for (int i = 1; i < count; ++i) {
        Candidate key = list[i];
        int j = i - 1;
        while (j >= 0 && list[j].dist > key.dist) {
            list[j + 1] = list[j];
            j = j - 1;
        }
        list[j + 1] = key;
    }
}

// =========================================================================
// Kernel: Warp-Centric Update with Heuristic & Optimizations
// =========================================================================
__global__ void apply_topology_updates_heuristic_v2(
    uint32_t* d_graph,               
    const float* d_dataset,          
    size_t dim,                      // [Change] 支持动态维度 (128, 512, 2048...)
    
    // --- Mailboxes ---
    const uint32_t* d_local_req_counts,
    const uint32_t* d_local_req_lists,
    uint32_t max_requests_local,
    
    const uint32_t* d_remote_req_counts,
    const uint32_t* d_remote_req_lists,
    uint32_t max_requests_remote,

    size_t num_existing,
    uint32_t total_degree,
    uint32_t local_degree
) {
    // 1. Warp ID 计算
    int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_local_id  = threadIdx.x / WARP_SIZE; // Block 内的 Warp 序号

    if (warp_global_id >= num_existing) return;
    
    uint32_t my_id = warp_global_id;
    uint32_t* my_row = d_graph + (size_t)my_id * total_degree;

    // -----------------------------------------------------------
    // Shared Memory 布局规划
    // -----------------------------------------------------------
    // 每个 Warp 需要:
    // 1. my_vec_s: float[dim]
    // 2. candidates: Candidate[MAX_CANDIDATES]
    // 3. results:    Candidate[64] (最大 degree)
    
    extern __shared__ char smem[];
    char* warp_smem_base = smem + warp_local_id * (
        dim * sizeof(float) + 
        MAX_CANDIDATES * sizeof(Candidate) + 
        64 * sizeof(Candidate) 
    ); // 注意对齐，这里简化写

    // 指针分配
    float* my_vec_s = (float*)warp_smem_base;
    Candidate* my_candidates = (Candidate*)(my_vec_s + dim);
    Candidate* my_results    = my_candidates + MAX_CANDIDATES;

    // -----------------------------------------------------------
    // Step 0: 预加载自己的向量到 SMEM (Coalesced Load)
    // -----------------------------------------------------------
    const float* g_my_vec = d_dataset + (size_t)my_id * dim;
    for (int i = lane_id; i < dim; i += 32) {
        my_vec_s[i] = g_my_vec[i];
    }
    // 注意：Warp 内无需 __syncthreads()，指令发射顺序保证写完后读

    // -----------------------------------------------------------
    // 循环处理 Local (pass 0) 和 Remote (pass 1)
    // -----------------------------------------------------------
    uint32_t offset_start = 0;
    uint32_t max_degree = local_degree;
    
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            offset_start = local_degree;
            max_degree = total_degree - local_degree;
        }

        // =======================================================
        // Optimization: 检查 Mailbox 是否为空
        // =======================================================
        uint32_t req_count = (pass == 0) ? d_local_req_counts[my_id] : d_remote_req_counts[my_id];
        
        // 如果没人想连我，且这是 Local/Remote 更新，
        // 说明现有的结构已经是“最好的”了（或者无需变动），直接跳过！
        // 这样可以避免昂贵的距离计算和裁剪过程。
        if (req_count == 0) {
            continue; 
        }

        uint32_t max_req = (pass == 0) ? max_requests_local : max_requests_remote;
        if (req_count > max_req) req_count = max_req;
        
        const uint32_t* req_list_base = (pass == 0) ? d_local_req_lists : d_remote_req_lists;
        const uint32_t* req_list = req_list_base + (size_t)my_id * max_req;

        // 保护现有邻居的strong区域
        // 90% 的概率保护， 10% 的概率放开
        uint32_t rng_state = wang_hash_insert(my_id + pass * 123456789);
        rng_state = wang_hash_insert(rng_state);
        bool protect_strong = (rng_state % 100) < 90 ? 1 : 0;

        uint32_t weak_start = protect_strong ? max_degree / 2 : 0;
        offset_start += weak_start;
        max_degree -= weak_start;

        // -------------------------------------------------------
        // Step 1: 收集所有候选者 (Current Neighbors + Mailbox)
        // -------------------------------------------------------
        int cand_count = 0; // 仅 Lane 0 维护
        
        // 1.A: 收集现有邻居
        for (int i = 0; i < max_degree; ++i) {
            uint32_t nid = my_row[offset_start + i];
            if (nid == 0xFFFFFFFF) continue;
            
            // 广播 ID
            nid = __shfl_sync(0xFFFFFFFF, nid, 0);

            // 计算 dist(My_SMEM, Neighbor_Global)
            const float* n_vec = d_dataset + (size_t)nid * dim;
            float d = calc_dist_smem_global(my_vec_s, n_vec, dim);
            if (req_list[0] - nid < 10000) {d *= 0.8;}                     // trick: 如果当前邻居中的节点和mailbox请求中的节点ID接近，说明可能是同一批次插入的点，距离打8折，优化裁剪效果

            if (lane_id == 0) {
                my_candidates[cand_count].id = nid;
                my_candidates[cand_count].dist = d;
                cand_count++;
            }
        }

        // 1.B: 收集 Mailbox 请求
        for (int i = 0; i < req_count; ++i) {
            uint32_t req_id;
            if (lane_id == 0) req_id = req_list[i];
            req_id = __shfl_sync(0xFFFFFFFF, req_id, 0);
            
            // 简单去重逻辑 (可选，略过)

            const float* req_vec = d_dataset + (size_t)req_id * dim;
            float d = calc_dist_smem_global(my_vec_s, req_vec, dim);

            if (lane_id == 0 && cand_count < MAX_CANDIDATES) {
                my_candidates[cand_count].id = req_id;
                my_candidates[cand_count].dist = d * 0.2;       // trick: Mailbox中的点我们认为更重要，距离打6折，优化后续的裁剪效果
                cand_count++;
            }
        }

        // -------------------------------------------------------
        // Step 2: 排序 (Sort)
        // -------------------------------------------------------
        if (lane_id == 0) {
            sort_candidates_lane0(my_candidates, cand_count);
        }
        __syncwarp(); 

        cand_count = __shfl_sync(0xFFFFFFFF, cand_count, 0);

        // DEBUG 输出候选列表
        // if (lane_id == 0 && warp_global_id == 1909 && pass == 1) {
        //     printf("Debug: Warp %d Pass %d Candidate List (count=%d):\n", warp_global_id, pass, cand_count);
        //     for (int i = 0; i < cand_count; ++i) {
        //         printf("  Cand %d: ID=%u Dist=%.4f\n", i, my_candidates[i].id, my_candidates[i].dist);
        //     }
        // }

        // -------------------------------------------------------
        // Step 3: 启发式裁减 (Heuristic Pruning)
        // -------------------------------------------------------
        int result_count = 0;

        // 遍历所有候选者
        for (int i = 0; i < cand_count && result_count < max_degree; ++i) {
            uint32_t c_id;
            float c_dist_n;
            if (lane_id == 0) {
                c_id = my_candidates[i].id;
                c_dist_n = my_candidates[i].dist;
            }
            c_id = __shfl_sync(0xFFFFFFFF, c_id, 0);
            c_dist_n = __shfl_sync(0xFFFFFFFF, c_dist_n, 0);

            // 这里的优化：
            // 我们不能把 C 放入 SMEM，因为空间不够。
            // 但我们可以把 C 的向量在 Loop 里加载到 寄存器 (如果 dim 小)
            // 或者：直接读 Global C vs Global R。
            // 鉴于 dim 可能很大(2048)，寄存器放不下，我们采用 Global-to-Global 的计算
            // 虽然慢点，但是安全。
            const float* c_vec_g = d_dataset + (size_t)c_id * dim;

            bool is_occluded = false;

            // 检查与 results 中已存在节点的距离
            for (int r = 0; r < result_count; ++r) {
                uint32_t r_id;
                if (lane_id == 0) r_id = my_results[r].id;
                r_id = __shfl_sync(0xFFFFFFFF, r_id, 0);

                const float* r_vec_g = d_dataset + (size_t)r_id * dim;

                int lane = threadIdx.x % 32;
                float dist_c_r = 0.0f;

                // 线程同步
                __syncwarp();

                if (dim == 1024) dist_c_r = cagra::device::calc_l2_dist_1024(c_vec_g, r_vec_g);
                else if (dim == 2048) dist_c_r = cagra::device::calc_l2_dist_2048(c_vec_g, r_vec_g);
                else if (dim == 128)  dist_c_r = cagra::device::calc_l2_dist_128(c_vec_g, r_vec_g);
                else if (dim == 960)  dist_c_r = cagra::device::calc_l2_dist_960(c_vec_g, r_vec_g);
                else if (dim == 96)  dist_c_r = cagra::device::calc_l2_dist_96(c_vec_g, r_vec_g);
                else printf("Error: Unsupported dim %d for dist_c_r calculation.\n", dim);

                if (dist_c_r < c_dist_n) {
                    is_occluded = true;
                    // if (lane_id == 0 && pass == 1 && warp_global_id == 1909) {
                    //     // Debug 输出遮挡信息
                    //     printf("Debug: Pass %d Y is %d Candidate c_id=%u dist=%.4f is occluded by r_id=%u dist=%.4f (dist_c_r=%.4f, and dist_c_n=%.4f)\n", 
                    //         pass, my_id, c_id, c_dist_n, r_id, my_results[r].dist, dist_c_r, c_dist_n);
                    // }
                    break;
                }
            }

            if (!is_occluded) {
                // 未被遮挡，通过广播同步状态
                if (__shfl_sync(0xFFFFFFFF, is_occluded ? 1 : 0, 0) == 0) { 
                    if (lane_id == 0) {
                        my_results[result_count].id = c_id;
                        my_results[result_count].dist = c_dist_n; // 存下来备用虽然后面没用到

                        // if (pass == 1 && warp_global_id == 1909) printf("Debug: Pass %d Y is %d Accepted c_id=%u dist=%.4f into results at pos %d\n", pass, my_id, c_id, c_dist_n, result_count);
                        
                        my_candidates[i].dist = -1.0f; // 标记为已选
                    }
                }
                // 所有线程更新
                result_count++;
            } else {
                
            }
        }

        // -------------------------------------------------------
        // Step 3.5: 填充剩余的结果到result
        // ------------------------------------------------------
        for (int tt = 0; tt < cand_count && result_count < max_degree; ++tt) {
            if (my_candidates[tt].dist < 0.0f) continue; // 已选过的跳过
            // if (lane_id == 0) {
            //     if (pass == 1 && warp_global_id == 1909) printf("Padding result with candidate id=%u dist=%.4f\n", my_candidates[tt].id, my_candidates[tt].dist);
            //     my_results[result_count].id = my_candidates[tt].id;
            //     my_results[result_count].dist = my_candidates[tt].dist;
            //     result_count++;
            // }
        }

        __syncwarp();
        // 输出一下final的result列表
        // if (lane_id == 0 && warp_global_id == 1909 && pass == 1) {
        //     printf("Debug: Warp %d Pass %d Final Result List (count=%d):\n", warp_global_id, pass, result_count);
        //     for (int i = 0; i < result_count; ++i) {
        //         printf("  Res %d: ID=%u Dist=%.4f\n", i, my_results[i].id, my_results[i].dist);
        //     }
        // }

        // -------------------------------------------------------
        // Step 4: 写回 Global Memory
        // -------------------------------------------------------
        // 将 my_results 写回 my_row warp 同步写回
        for (int i = 0; i < max_degree; i += WARP_SIZE) {
            if (i < result_count) {
                my_row[offset_start + i] = my_results[i].id;
            }
        }
    
    }
}

void launch_update_v2(
    uint32_t* d_graph,               
    const float* d_dataset,          
    size_t dim,                      
    
    // --- Mailboxes ---
    const uint32_t* d_local_req_counts,
    const uint32_t* d_local_req_lists,
    uint32_t max_requests_local,
    
    const uint32_t* d_remote_req_counts,
    const uint32_t* d_remote_req_lists,
    uint32_t max_requests_remote,

    size_t num_existing,
    uint32_t total_degree,
    uint32_t local_degree
) {
    // printf("Launching apply_topology_updates_heuristic_v2 with dim=%zu\n", dim);
    int block_size = 256;
    if (dim > 1024) block_size = 128;    // 避免shared_memory爆炸
    
    int warps_per_block = block_size / WARP_SIZE;
    size_t smem_per_warp =  dim * sizeof(float) +
                        MAX_CANDIDATES * sizeof(Candidate) +        // 候选列表大小
                        64 * sizeof(Candidate);                     // 结果列表大小（我们的local和remote都不会超过64）
    smem_per_warp = (smem_per_warp + 15) & ~15;                     // 16字节对齐

    size_t total_smem = warps_per_block * smem_per_warp;

    size_t total_threads = num_existing * WARP_SIZE;
    size_t grid_size = (total_threads + block_size - 1) / block_size;

    apply_topology_updates_heuristic_v2<<<grid_size, block_size, total_smem>>>(
        d_graph,
        d_dataset,
        dim,
        d_local_req_counts,
        d_local_req_lists,
        max_requests_local,
        d_remote_req_counts,
        d_remote_req_lists,
        max_requests_remote,
        num_existing,
        total_degree,
        local_degree
    );

    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------------------
// V2: 基于 HNSW 启发式策略的新节点填充
// -------------------------------------------------------------------------
__global__ void fill_new_nodes_heuristic_v2(
    uint32_t* d_graph,               // [Total_N, 32] (写入目标)
    const float* d_dataset,          // 原始向量 (用于算 Cand-Result 距离)
    size_t dim,
    const uint64_t* d_ts,            // 时间戳
    
    // KNN 结果 (必须包含距离!)
    int64_t* d_local_knn,      // [num_new, search_k_local]
    const float* d_local_dists,      // [num_new, search_k_local] (新增)
    
    int64_t* d_global_knn,     // [num_new, search_k_global]
    const float* d_global_dists,     // [num_new, search_k_global] (新增)

    size_t num_existing,
    size_t num_new,
    uint32_t total_degree,
    uint32_t local_degree,
    uint32_t search_k_local,
    uint32_t search_k_global
) {
    // 1. Warp ID 计算 (Warp-per-Node)
    int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_local_id  = threadIdx.x / WARP_SIZE;

    if (warp_global_id >= num_new) return;

    // 当前新节点的全局 ID
    size_t my_global_id = num_existing + (size_t)warp_global_id;
    uint64_t my_ts = d_ts[my_global_id];
    uint32_t* my_graph_row = d_graph + my_global_id * total_degree;

    // -----------------------------------------------------------
    // SMEM 分配
    // -----------------------------------------------------------
    // 需要存: MyVector + Local Results(Temp) + Global Results(Temp)
    // 这里我们直接复用 Update Kernel 的逻辑：
    // 每次处理一个 Pass (Local/Remote)，处理完直接写入 Global Memory
    
    extern __shared__ char smem[];
    char* warp_smem_base = smem + warp_local_id * (
        dim * sizeof(float) + 
        MAX_CANDIDATES * sizeof(Candidate) + // Candidates (来自 KNN)
        64 * sizeof(Candidate)               // Results
    );

    float* my_vec_s = (float*)warp_smem_base;
    Candidate* my_candidates = (Candidate*)(my_vec_s + dim);
    Candidate* my_results    = my_candidates + MAX_CANDIDATES;

    // 预加载我的向量 (Coalesced Load)
    const float* g_my_vec = d_dataset + (size_t)my_global_id * dim;
    for (int i = lane_id; i < dim; i += 32) {
        my_vec_s[i] = g_my_vec[i];
    }
    // 注意：如果是同一个 warp 内，不需要 sync，指令流同步

    // ===========================================================
    // Two Passes: 0 -> Local, 1 -> Remote
    // ===========================================================
    
    for (int pass = 0; pass < 2; ++pass) {
        uint32_t max_degree = (pass == 0) ? local_degree : (total_degree - local_degree);
        uint32_t search_k = (pass == 0) ? search_k_local : search_k_global;
        int64_t* knn_indices = (pass == 0) ? d_local_knn : d_global_knn;
        const float* knn_dists = (pass == 0) ? d_local_dists : d_global_dists;
        uint32_t offset_start = (pass == 0) ? 0 : local_degree;

        // 指针偏移
        knn_indices += (size_t)warp_global_id * search_k;
        knn_dists += (size_t)warp_global_id * search_k;

        // -------------------------------------------------------
        // Step 1: 收集候选者 (从 KNN 结果中读取)
        // -------------------------------------------------------
        int cand_count = 0; // Lane 0 Maintain

        // 直接遍历 KNN 结果
        // 因为 KNN 结果通常是有序的 (dist 小 -> 大)，我们直接按顺序读进来即可
        // 但需要过滤无效点、自环、以及 Local Pass 的 TS 约束
        
        // 算了，直接在 Loop 里 Lane 0 读吧，search_k 也就 128，几十个 cycle 的事。
        if (lane_id == 0) {
            for (int k = 0; k < search_k; ++k) {
                if (cand_count >= MAX_CANDIDATES) break;
                
                int64_t idx = knn_indices[k];
                float d = knn_dists[k];
                
                if (idx < 0) continue;
                uint32_t uid = (uint32_t)idx;
                if (uid == my_global_id) continue;
                
                if (pass == 0) {
                    knn_indices[k] = -1; // Mark as used
                    my_candidates[cand_count].id = uid;
                    my_candidates[cand_count].dist = d;
                    cand_count++;
                } else {
                    int64_t cand_ts = d_ts[uid];
                    if (cand_ts != my_ts) {
                        knn_indices[k] = -1; // Mark as used
                        my_candidates[cand_count].id = uid;
                        my_candidates[cand_count].dist = d;
                        cand_count++;
                    }
                }
            }

            for (int k = 0; k < search_k && cand_count < MAX_CANDIDATES; ++k) {
                int64_t idx = knn_indices[k];
                if (idx < 0) continue; // Already used
                
                uint32_t uid = (uint32_t)idx;
                if (uid == my_global_id) continue;
                
                float d = knn_dists[k];
                my_candidates[cand_count].id = uid;
                my_candidates[cand_count].dist = d;
                cand_count++;
            }
        }
        
        // // DEBUG 输出候选列表的前32个
        // if (lane_id == 0 && warp_global_id == 0) {
        //     for (int i = 0; i < cand_count; ++i) {
        //         uint32_t c_id;
        //         float c_dist;
        //         c_id = my_candidates[i].id;
        //         c_dist = my_candidates[i].dist;                
        //         printf("Debug: Warp %d Pass %d Candidate %d: ID=%u Dist=%.4f\n", warp_global_id, pass, i, c_id, c_dist);
        //     }
        // }


        // 【关键同步】
        __syncwarp();
        cand_count = __shfl_sync(0xFFFFFFFF, cand_count, 0);

        // -------------------------------------------------------
        // Step 2: 启发式裁减 (Heuristic Pruning)
        // -------------------------------------------------------
        // 代码完全复用 Update Kernel 的逻辑
        int result_count = 0;

        for (int i = 0; i < cand_count; ++i) {
            // 检查已满
            // if (i == 0 && lane_id == 0) {
            //     printf("Debug: Warp %d Pass %d Starting Heuristic Pruning with cand_count=%d\n", warp_global_id, pass, cand_count);
            // }
            int current_res_count = __shfl_sync(0xFFFFFFFF, result_count, 0);
            if (current_res_count >= max_degree) break;

            uint32_t c_id;
            float c_dist_n;
            if (lane_id == 0) {
                c_id = my_candidates[i].id;
                c_dist_n = my_candidates[i].dist;
            }
            c_id = __shfl_sync(0xFFFFFFFF, c_id, 0);
            c_dist_n = __shfl_sync(0xFFFFFFFF, c_dist_n, 0);

            // 过滤无效
            if (c_id == 0xFFFFFFFF) continue; // Should not happen

            // 算距离需要的指针
            const float* c_vec_g = d_dataset + (size_t)c_id * dim;
            bool is_occluded = false;

            for (int r = 0; r < current_res_count; ++r) {
                uint32_t r_id;
                if (lane_id == 0) r_id = my_results[r].id;
                r_id = __shfl_sync(0xFFFFFFFF, r_id, 0);

                const float* r_vec_g = d_dataset + (size_t)r_id * dim;
                float dist_c_r = 0.0f;

                // Call Dist Func
                if (dim == 128) dist_c_r = cagra::device::calc_l2_dist_128(c_vec_g, r_vec_g);
                else if (dim == 1024) dist_c_r = cagra::device::calc_l2_dist_1024(c_vec_g, r_vec_g);
                else if (dim == 2048) dist_c_r = cagra::device::calc_l2_dist_2048(c_vec_g, r_vec_g);
                else if (dim == 960) dist_c_r = cagra::device::calc_l2_dist_960(c_vec_g, r_vec_g);
                else printf("Error: Unsupported dim %d for dist_c_r calculation.\n", dim);

                if (dist_c_r < c_dist_n) {
                    is_occluded = true;
                    // if (lane_id == 0 && pass == 0 && warp_global_id == 0) {
                    //     // Debug 输出遮挡信息
                    //     printf("Debug: Pass %d Y is %lu Candidate c_id=%u dist=%.4f is occluded by r_id=%u dist=%.4f (dist_c_r=%.4f, and dist_c_n=%.4f)\n", 
                    //         pass, my_global_id, c_id, c_dist_n, r_id, my_results[r].dist, dist_c_r, c_dist_n);
                    // }
                    break;
                }
            }
            
            int vote = __shfl_sync(0xFFFFFFFF, is_occluded?1:0, 0);
            if (vote == 0) {
                if (lane_id == 0) {
                    my_results[result_count].id = c_id;
                    my_results[result_count].dist = c_dist_n;
                    result_count++;
                    // if (pass == 0 && warp_global_id == 0) {
                    //     printf("Debug: Pass %d Y is %lu Accepted c_id=%u dist=%.4f into results at pos %d\n", pass, my_global_id, c_id, c_dist_n, result_count);
                    // }
                }
                result_count = __shfl_sync(0xFFFFFFFF, result_count, 0);
            }
        }

        // -------------------------------------------------------
        // Step 3: 回填备胎 (Backfilling)
        // -------------------------------------------------------
        // 如果没填满，从 my_candidates 里捞剩下的
        int current_res_count = __shfl_sync(0xFFFFFFFF, result_count, 0);
        
        if (current_res_count < max_degree) {
            if (lane_id == 0) {
                for(int i=0; i<cand_count && result_count < max_degree; ++i) {
                    uint32_t c_id = my_candidates[i].id;
                    // Check existence
                    bool exists = false;
                    for(int r=0; r<result_count; ++r) {
                        if (my_results[r].id == c_id) { exists = true; break; }
                    }
                    if (!exists) {
                        my_results[result_count].id = c_id;
                        result_count++;
                        // printf("Paddiing into result pos %d with candidate id=%u dist=%.4f\n", result_count-1, c_id, my_candidates[i].dist);
                    }
                }
            }
            result_count = __shfl_sync(0xFFFFFFFF, result_count, 0);
        }

        // -------------------------------------------------------
        // Step 4: 写回 Global Memory
        // -------------------------------------------------------
        for (int i = lane_id; i < max_degree; i += WARP_SIZE) {
            uint32_t val = 0xFFFFFFFF;
            if (i < result_count) {
                val = my_results[i].id;
            }
            my_graph_row[offset_start + i] = val;
        }
    }
}


void update_topology_gpu_opt(
    uint32_t* d_graph,              // [In/Out] 全量图
    const float* d_dataset,         // [In] 全量数据集 (用于距离计算)
    size_t dim,                     // 向量维度
    size_t target_ts,               // 目标时间戳
    const uint64_t* d_ts,           // [In] 时间戳 (Fill阶段需要)
    int64_t* d_search_indices,// [In] 搜索结果
    float* d_search_dists,    // [In] 搜索距离
    int64_t* d_search_global,    // [In] 全局搜索结果 (优化版可共用)
    float* d_search_global_dists, // [In] 全局搜索距离
    size_t num_existing,            // 老节点数量
    size_t num_new,                 // 新节点数量
    uint32_t total_degree,          // 32
    uint32_t local_degree,          // 28
    uint32_t search_k_local,               // 128
    uint32_t search_k_global,               // 128
    bool use_heuristic
) {
    if (num_new == 0) return;

    // -------------------------------------------------------------
    // Step 1: 填充新节点的出边 (Fill Outbound)
    // -------------------------------------------------------------
    // Grid: 基于新节点数量


    // 假设 d_global_knn 和 d_local_knn 共用 d_search_indices
    bool use_v1 = true;
    int block_size = 256;
    int grid_size_new = (num_new + block_size - 1) / block_size;
    if (use_v1) {    
        fill_new_nodes_kernel_opt<<<grid_size_new, block_size>>>(
            d_graph,
            d_ts,
            d_search_indices, // local knn
            d_search_global, // global knn
            num_existing,
            num_new,
            total_degree,
            local_degree,
            search_k_local,
            search_k_global
        );
    } else {
        // printf("Use fill_new_nodes_heuristic_v2 with dim=%zu\n", dim);
        int block_size = 256;
        if (dim > 1024) block_size = 128;    // 避免shared_memory爆炸
        int total_threads = num_new * WARP_SIZE;
        int grid_size_new = (total_threads + block_size - 1) / block_size;

        size_t smem_per_warp =  dim * sizeof(float) +
                            MAX_CANDIDATES * sizeof(Candidate) +        // 候选列表大小
                            64 * sizeof(Candidate);                     // 结果列表大小（我们的local和remote都不会超过64）
        smem_per_warp = (smem_per_warp + 15) & ~15;                     // 16字节对齐

        int warps_per_block = block_size / WARP_SIZE;
        size_t total_smem = warps_per_block * smem_per_warp;

        fill_new_nodes_heuristic_v2<<<grid_size_new, block_size, total_smem>>>(
            d_graph,
            d_dataset,
            dim,
            d_ts,
            d_search_indices, // local knn
            d_search_dists,
            d_search_global, // global knn
            d_search_global_dists,
            num_existing,
            num_new,
            total_degree,
            local_degree,
            search_k_local,
            search_k_global
        );
    }

    CUDA_CHECK(cudaGetLastError());
    // 这里不需要 Sync，因为 Step 2 依赖 Step 1 的结果，同一个流内会自动串行

    // -------------------------------------------------------------
    // Step 2: 准备 Mailbox (中间存储)
    // -------------------------------------------------------------
    uint32_t local_max_requests = local_degree / 2;
    uint32_t remote_max_requests = (total_degree - local_degree) / 2;
    
    // 申请显存 (只针对老节点 num_existing)
    uint32_t *d_local_req_counts, *d_local_req_lists;
    uint32_t *d_remote_req_counts, *d_remote_req_lists;

    size_t counts_size = (num_existing + num_new) * sizeof(uint32_t);
    size_t local_lists_size = (num_existing + num_new) * local_max_requests * sizeof(uint32_t);
    size_t remote_lists_size = (num_existing + num_new) * remote_max_requests * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc(&d_local_req_counts, counts_size));
    CUDA_CHECK(cudaMalloc(&d_local_req_lists, local_lists_size));
    CUDA_CHECK(cudaMalloc(&d_remote_req_counts, counts_size));
    CUDA_CHECK(cudaMalloc(&d_remote_req_lists, remote_lists_size));

    // 初始化计数器为 0
    CUDA_CHECK(cudaMemset(d_local_req_counts, 0, counts_size));
    CUDA_CHECK(cudaMemset(d_remote_req_counts, 0, counts_size));

    // -------------------------------------------------------------
    // Step 3: 生成请求 (Generate Requests)
    // -------------------------------------------------------------
    // Grid: 基于新节点数量 (因为是新节点发起请求)
    // 使用优化版 Kernel (无需查 d_ts，直接根据 index 判断 local/remote)
    
    generate_update_requests_kernel_opt<<<grid_size_new, block_size>>>(
        d_graph,
        // d_ts, // 不需要了
        d_local_req_counts, 
        d_local_req_lists,
        d_remote_req_counts, 
        d_remote_req_lists,
        num_existing,
        num_new,
        target_ts,
        d_ts,
        total_degree,
        local_degree,
        local_max_requests,
        remote_max_requests
    );
    CUDA_CHECK(cudaGetLastError());

    // -------------------------------------------------------------
    // Step 4: 应用更新 (Apply Updates)
    // -------------------------------------------------------------
    // Grid: 基于老节点数量 (因为是老节点处理收件箱)
    if (use_v1) {    
        int grid_size_existing = (num_existing + block_size - 1) / block_size;

        apply_topology_updates_kernel_opt<<<grid_size_existing, block_size>>>(
            d_graph,
            d_local_req_counts, 
            d_local_req_lists, 
            local_max_requests, // local max
            d_remote_req_counts, 
            d_remote_req_lists, 
            remote_max_requests, // remote max
            d_ts,
            num_existing,
            total_degree,
            local_degree
        );
    } else {
        launch_update_v2(
            d_graph,
            d_dataset,
            dim,
            d_local_req_counts, 
            d_local_req_lists, 
            local_max_requests, // local max
            d_remote_req_counts, 
            d_remote_req_lists, 
            remote_max_requests, // remote max
            num_existing + num_new,
            total_degree,
            local_degree
        );
    }

    CUDA_CHECK(cudaGetLastError());
    
    // 等待所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // Step 5: 清理资源
    // -------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_local_req_counts));
    CUDA_CHECK(cudaFree(d_local_req_lists));
    CUDA_CHECK(cudaFree(d_remote_req_counts));
    CUDA_CHECK(cudaFree(d_remote_req_lists));
}

// =============================================================================
// Refine Kernel: 计算精确距离并重排序 (针对 Insert 阶段)
// =============================================================================
// 假设 K 是 2 的幂次 (64, 128, 256...)
// 每个线程处理 N = K / 32 个候选
template <int N>
__global__ void refine_cagra_candidates_kernel(
    const float* d_dataset,         // [num_existing, dim] 老数据集
    const float* d_queries,         // [num_new, dim] 新插入的数据(作为Query)
    int64_t* d_indices,             // [num_new, K] 输入/输出索引
    float* d_dists,                 // [num_new, K] 输入/输出距离
    size_t num_existing,
    size_t num_new,
    uint32_t dim,
    uint32_t K
) {
    // 1. 计算当前 Warp 负责的 Query ID
    // 假设 BlockDim = 256 (8 Warps)
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t lane_id = threadIdx.x % 32;

    if (warp_id >= num_new) return;

    // 当前 Query 向量指针
    const float* query_vec = d_queries + warp_id * dim;
    
    // 当前 Query 在 indices/dists 数组中的偏移
    size_t row_offset = warp_id * K;

    // 2. 寄存器数组：存储分给当前线程的候选点
    float my_dists[N];
    uint32_t my_indices[N];

    // 初始化
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        my_dists[i] = 3.40282e38f; // MAX_FLOAT
        my_indices[i] = 0xFFFFFFFF;
    }

    // 3. 加载候选点并计算精确距离 (Warp 协作)
    // --------------------------------------------------------
    // 我们需要遍历 K 个候选。
    // 为了利用 calc_l2_dist_1024 (Warp级算子)，我们必须所有线程同步处理同一个候选。
    
    for (int k = 0; k < K; ++k) {
        // 读取索引 (广播读取，大家读一样的)
        // 注意：d_indices 是 int64，但我们内部处理用 uint32，最后再转回
        int64_t idx_64 = d_indices[row_offset + k];
        
        float dist = 3.40282e38f;

        // 验证索引合法性
        if (idx_64 >= 0 && idx_64 < num_existing) {
            uint32_t idx_32 = (uint32_t)idx_64;
            const float* cand_vec = d_dataset + (size_t)idx_32 * dim;

            // Warp 协作计算精确 L2
            float dist = 0.0f;
            if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_vec, cand_vec);
            else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_vec, cand_vec);
            else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_vec, cand_vec);
            else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_vec, cand_vec);
            else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_vec, cand_vec);
            else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_vec, cand_vec);
            else {
                // 对于非特殊维度，调用通用版本
                printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
            }
        }

        // 分发结果到对应的寄存器
        // 候选 k 属于线程 (k % 32) 的第 (k / 32) 个槽位
        int owner_lane = k % 32;
        int reg_idx = k / 32;

        if (lane_id == owner_lane) {
            my_dists[reg_idx] = dist;
            my_indices[reg_idx] = (uint32_t)idx_64; // 暂时截断为 u32 用于排序
        }
    }

    // 4. 双调排序 (Warp Sort)
    // --------------------------------------------------------
    // 升序排列：距离小的在前
    cagra::bitonic::warp_sort<float, uint32_t, N>(my_dists, my_indices, true);

    // 5. 写回 Global Memory
    // --------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int k = i * 32 + lane_id;
        // 写回时转回 int64
        if (k < K) {
            uint32_t idx = my_indices[i];
            d_indices[row_offset + k] = (idx == 0xFFFFFFFF) ? -1 : (int64_t)idx;
            d_dists[row_offset + k] = my_dists[i];
        }
    }
}

// Host Wrapper
void refine_cagra_candidates(const float* d_dataset,
                             const float* d_queries,
                             int64_t* d_indices,
                             float* d_dists,
                             size_t num_existing,
                             size_t num_new,
                             uint32_t dim,
                             uint32_t k)
{
    // std::cout << ">> [CAGRA Algo] Refining " << num_new << " queries (K=" << k << ")..." << std::endl;

    int block_size = 256; // 8 Warps per block
    int warps_per_block = 8;
    int grid_size = (num_new + warps_per_block - 1) / warps_per_block;

    // 根据 K 选择模板 N
    if (k <= 64) {
        refine_cagra_candidates_kernel<2><<<grid_size, block_size>>>(
            d_dataset, d_queries, d_indices, d_dists, num_existing, num_new, dim, k);
    } else if (k <= 128) {
        refine_cagra_candidates_kernel<4><<<grid_size, block_size>>>(
            d_dataset, d_queries, d_indices, d_dists, num_existing, num_new, dim, k);
    } else if (k <= 256) {
        refine_cagra_candidates_kernel<8><<<grid_size, block_size>>>(
            d_dataset, d_queries, d_indices, d_dists, num_existing, num_new, dim, k);
    } else {
        // Fallback for larger K (512)
        refine_cagra_candidates_kernel<16><<<grid_size, block_size>>>(
            d_dataset, d_queries, d_indices, d_dists, num_existing, num_new, dim, k);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


} // namespace cagra