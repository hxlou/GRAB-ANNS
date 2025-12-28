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
    const int64_t* d_local_knn,     // [num_new, search_k_local]
    const int64_t* d_global_knn,    // [num_new, search_k_global]
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
    const int64_t* my_local_candidates = d_local_knn + tid * search_k_local;

    for (uint32_t k = 0; k < search_k_local; ++k) {
        if (filled_local >= local_degree) break;

        int64_t cand_id_64 = my_local_candidates[k];
        uint32_t cand_id = (uint32_t)cand_id_64;

        // 过滤无效值
        if (cand_id_64 < 0) continue;
        if (cand_id == my_global_id) continue; // 不连自己
        if (cand_id >= 0x7fffffff) continue; // 过滤异常大 ID

        // 强制时间戳约束：必须同桶
        if (d_ts[cand_id] == my_ts) {
            my_graph_row[filled_local++] = (uint32_t)cand_id;
        }
    }

    // 补齐 Local
    while (filled_local < local_degree) {
        my_graph_row[filled_local++] = 0xFFFFFFFF;
    }

    // -------------------------------------------------------------
    // Phase 2: 填充 Remote Edges [local_degree, total_degree-1]
    // -------------------------------------------------------------
    int filled_remote = 0;
    int max_remote = total_degree - local_degree;
    const int64_t* my_global_candidates = d_global_knn + tid * search_k_global;

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
        }
    }

    // 补齐 Remote
    while (filled_remote < max_remote) {
        my_graph_row[local_degree + filled_remote] = 0xFFFFFFFF;
        filled_remote++;
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
            rng_state = wang_hash_insert(rng_state);
            bool attack_strong = (rng_state % 100) < PROB_THRESHOLD;

            // 2. 确定攻击范围
            uint32_t target_start = attack_strong ? strong_start : weak_start;
            uint32_t target_len   = attack_strong ? strong_len : weak_len;

            // 3. 随机替换
            if (target_len > 0) {
                rng_state = wang_hash_insert(rng_state);
                int slot = target_start + (rng_state % target_len);
                my_row[slot] = candidate;
            }
        }
    }
}


void update_topology_gpu_opt(
    uint32_t* d_graph,              // [In/Out] 全量图
    const uint64_t* d_ts,           // [In] 时间戳 (Fill阶段需要)
    const int64_t* d_search_indices,// [In] 搜索结果
    const int64_t* d_search_global,    // [In] 全局搜索结果 (优化版可共用)
    size_t num_existing,            // 老节点数量
    size_t num_new,                 // 新节点数量
    uint32_t total_degree,          // 32
    uint32_t local_degree,          // 28
    uint32_t search_k_local,               // 128
    uint32_t search_k_global               // 128
) {
    if (num_new == 0) return;

    // -------------------------------------------------------------
    // Step 1: 填充新节点的出边 (Fill Outbound)
    // -------------------------------------------------------------
    // Grid: 基于新节点数量
    int block_size = 256;
    int grid_size_new = (num_new + block_size - 1) / block_size;

    // 你的 fill kernel 名字叫 fill_new_nodes_kernel_opt
    // 假设 d_global_knn 和 d_local_knn 共用 d_search_indices
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

    size_t counts_size = num_existing * sizeof(uint32_t);
    size_t local_lists_size = num_existing * local_max_requests * sizeof(uint32_t);
    size_t remote_lists_size = num_existing * remote_max_requests * sizeof(uint32_t);

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
    int grid_size_existing = (num_existing + block_size - 1) / block_size;

    apply_topology_updates_kernel_opt<<<grid_size_existing, block_size>>>(
        d_graph,
        d_local_req_counts, 
        d_local_req_lists, 
        local_max_requests, // local max
        d_remote_req_counts, 
        d_remote_req_lists, 
        remote_max_requests, // remote max
        num_existing,
        total_degree,
        local_degree
    );
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