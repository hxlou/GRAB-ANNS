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

__device__ unsigned long long get_global_time() {
    unsigned long long global_timer;
    // 使用内联汇编读取特殊寄存器 %globaltimer
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_timer));
    return global_timer;
}

/**
 * @brief 选择下一轮的父节点 (Pickup Next Parents)
 * 
 * 逻辑：
 * 1. 遍历 internal_topk_indices (结果队列)。
 * 2. 找到那些 最高位(MSB)为0 的节点 (即未被访问过的节点)。
 * 3. 收集前 search_width 个这样的节点，存入 next_parent_indices。
 * 4. 将被选中的节点在原队列中标记为已访问 (MSB 置 1)。
 * 
 * 注意：此函数假定由一个 Warp (32线程) 调用。
 * 
 * @param terminate_flag        [Out] 终止标志 (如果找不到新节点则置 1)
 * @param next_parent_indices   [Out] 输出：下一轮要扩展的节点 ID 列表
 * @param internal_topk_indices [In/Out] 当前 TopK 队列 (会被原地修改标记位)
 * @param internal_topk_size    当前 TopK 队列的有效元素数量
 * @param search_width          需要选出多少个父节点 (任意正整数)
 */
__device__ __forceinline__ void pickup_next_parents(uint32_t* terminate_flag,
                                                    uint32_t* next_parent_indices,
                                                    uint32_t* internal_topk_indices,
                                                    uint32_t internal_topk_size,
                                                    uint32_t search_width)
{
    // 只允许 Warp 的前 32 个线程执行此逻辑
    // (通常调用者会保证只用 block 内的前 32 线程调这个)
    uint32_t lane_id = threadIdx.x % 32;
    
    // 最高位掩码 (用于标记 Visited)
    constexpr uint32_t MSB_MASK = 0x80000000;
    constexpr uint32_t INVALID_IDX = 0xFFFFFFFF;

    // 1. 初始化输出数组 (并行初始化)
    for (uint32_t i = lane_id; i < search_width; i += 32) {
        next_parent_indices[i] = INVALID_IDX;
    }

    uint32_t num_new_parents = 0; // 已收集到的父节点数量

// 2. 遍历 TopK 队列
    for (uint32_t j = threadIdx.x; j < internal_topk_size; j += 32) {
        
        uint32_t node_id = internal_topk_indices[j]; 
        int is_new_parent = 0;

        // 检查条件
        if ((node_id & MSB_MASK) == 0 && node_id != INVALID_IDX) {
            is_new_parent = 1;
        }

        // 3. Warp 投票 (获取全局状态)
        uint32_t vote_mask = __ballot_sync(0xffffffff, is_new_parent);

        // 4. 只有“我找到了”的人才需要计算排名和写入
        if (is_new_parent) {
            // 计算我在这个 Warp 的当前轮次里排第几
            uint32_t rank_in_warp = __popc(vote_mask & ((1u << threadIdx.x) - 1));
            
            // 计算全局写入位置
            uint32_t global_rank = num_new_parents + rank_in_warp;
            
            // 如果没满，就写入
            if (global_rank < search_width) {
                next_parent_indices[global_rank] = node_id;
                // 标记为已访问
                internal_topk_indices[j] |= MSB_MASK;
            }
        }

        // 5. 所有人都要更新计数器，保持同步
        num_new_parents += __popc(vote_mask);

        // 如果已经找够了，全员退出
        if (num_new_parents >= search_width) { 
            break; 
        }
    }

    // 7. 设置终止标志
    // 只有当完全没找到任何新节点时，才终止
    if (lane_id == 0) {
        if (num_new_parents == 0) {
            *terminate_flag = 1; // Terminate
        } else {
            *terminate_flag = 0; // Continue
        }
    }
}


template <int N> // N = Capacity / 32
__device__ __forceinline__ void load_sort_store(float* smem_dists, uint32_t* smem_indices, uint32_t capacity) {
    // 1. 定义寄存器
    float key[N];
    uint32_t val[N];
    int lane_id = threadIdx.x; // 0-31

    // 2. 从 Shared Memory 加载到寄存器
    // 【核心修复】必须使用 Blocked Layout 以匹配 warp_sort 的逻辑
    // Thread 0 读取 [0, 1, ..., N-1]
    // Thread 1 读取 [N, N+1, ..., 2N-1]
    // 公式: idx = lane_id * N + i
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id * N + i; 
        
        // 这里的 idx 必然 < capacity (因为 capacity = 32 * N)
        key[i] = smem_dists[idx];
        val[i] = smem_indices[idx];
    }

    // 3. 寄存器内双调排序
    // warp_sort 内部假设 key[i] 对应的全局索引就是 lane_id * N + i
    // 现在加载逻辑和排序逻辑终于对齐了！
    cagra::bitonic::warp_sort<float, uint32_t, N>(key, val, true); // true = 升序

    // 4. 写回 Shared Memory
    // 同样使用 Blocked Layout 写回，保证顺序一致
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id * N + i;
        smem_dists[idx] = key[i];
        smem_indices[idx] = val[i];
    }
}

// =============================================================================
// Search Kernel: CAGRA 搜索核心
// =============================================================================
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
    uint32_t queue_capacity     // 必须是 32 的倍数 (通常也是 2 的幂次)
) {

    // -------------------------------------------------------------
    // 1. Shared Memory 动态布局初始化
    // -------------------------------------------------------------
    extern __shared__ uint8_t smem[]; 

    size_t offset = 0;

    // A. Query Buffer
    float* query_buffer = (float*)(smem + offset);
    offset += (dim * sizeof(float) + 15) & ~15;

    // B. Hashmap
    uint32_t* visited_hash = (uint32_t*)(smem + offset);
    offset += ((1u << hash_bitlen) * sizeof(uint32_t) + 15) & ~15;

    // C. Result Queue
    uint32_t* result_indices = (uint32_t*)(smem + offset);
    offset += (queue_capacity * sizeof(uint32_t) + 15) & ~15;

    float* result_dists = (float*)(smem + offset);
    offset += (queue_capacity * sizeof(float) + 15) & ~15;

    // D. Parent List
    uint32_t* parent_list = (uint32_t*)(smem + offset);
    offset += (search_width * sizeof(uint32_t) + 15) & ~15;

    // E. Flags
    volatile uint32_t* terminate_flag = (uint32_t*)(smem + offset);

    // -------------------------------------------------------------
    // 2. 线程与任务分配
    // -------------------------------------------------------------
    const uint32_t query_id = blockIdx.x;
    if (query_id >= num_queries) return;

    const uint32_t tid = threadIdx.x;

    // 加载 Query
    const float* global_query = queries_ptr + (size_t)query_id * dim;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = global_query[i];
    }
    
    if (tid == 0) *terminate_flag = 0;
    cagra::hashmap::init(visited_hash, hash_bitlen);
    
    __syncthreads(); 
    // -------------------------------------------------------------
    // 3. 初始种子阶段
    // -------------------------------------------------------------
    uint32_t seed_offset = query_id * num_seeds_per_query;
    uint32_t* local_seed_ptr = nullptr;
    if (seed_ptr != nullptr) {
        local_seed_ptr = (uint32_t*)(seed_ptr + seed_offset);
        // 调用混合初始化函数
        cagra::device::compute_distance_to_init_nodes(
            result_indices,
            result_dists,
            query_buffer,
            dataset_ptr,
            num_dataset,
            dim,
            queue_capacity,
            num_seeds,          // 目标总种子数 (例如 32)
            local_seed_ptr,     // 当前 Query 的外部种子起始位置
            num_seeds_per_query, // 外部提供了多少个
            rand_xor_mask,
            visited_hash,
            hash_bitlen
        );
    } else {
        // 如果没有提供 seed ，直接随机初始化即可
        cagra::device::compute_distance_to_random_nodes(
            result_indices, result_dists, query_buffer, dataset_ptr,
            num_dataset, dim, queue_capacity, num_seeds, rand_xor_mask,
            visited_hash, hash_bitlen
        );
    }
    __syncthreads();

    // -------------------------------------------------------------
    // 4. 搜索主循环
    // -------------------------------------------------------------
    uint32_t iter = 0;
    uint32_t hash_reset_iter = 20; // 每隔这么多轮重置 Hashmap
    for (; iter < max_iterations; ++iter) {
        
        // 更新哈希表，清空并加入topk中的数据到哈希表中
        if (iter > 0 && (iter % hash_reset_iter == 0)) {
            // 1. 清空
            cagra::hashmap::init(visited_hash, hash_bitlen);
            __syncthreads();
            
            // 2. 恢复 (把 itopk 里的节点重新加回去)
            // 为什么只恢复 itopk？因为只有这些节点是“当前最优”，
            // 我们不希望重新计算它们的距离，也不希望重新扩展它们（如果已标记为 Parent）。
            cagra::hashmap::restore(visited_hash, hash_bitlen, result_indices, itopk_size);
            __syncthreads();
        }

        // --- Step A: 排序 (仅 Warp 0 工作) ---        
        if (tid < 32) {
            // Shadow variable issue: 之前代码这里定义了 local queue_capacity 导致错误
            // 这里我们使用传入的 queue_capacity 参数，并用 switch/if 处理模板参数 N
            
            if (queue_capacity == 64) {
                load_sort_store<2>(result_dists, result_indices, 64);
            } else if (queue_capacity == 128) {
                load_sort_store<4>(result_dists, result_indices, 128);
            } else if (queue_capacity == 256) {
                load_sort_store<8>(result_dists, result_indices, 256);
            } else if (queue_capacity == 512) {
                load_sort_store<16>(result_dists, result_indices, 512);
            } else if (queue_capacity == 1024) {
                load_sort_store<32>(result_dists, result_indices, 1024);
            } else if (queue_capacity == 2048) {
                load_sort_store<64>(result_dists, result_indices, 2048);
            } else if (queue_capacity == 4096) {
                load_sort_store<128>(result_dists, result_indices, 4096);
            } else if (queue_capacity == 8192) {
                load_sort_store<256>(result_dists, result_indices, 8192);
            } else {
                // 不支持的容量大小
                if (tid == 0) {
                    printf(">> [search_kernel] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
                }
            }
            // 如果更大，可以继续加 case
        }
        __syncthreads();

        // --- Step B: 选父节点 ---
        if (tid < 32) {
            cagra::device::pickup_next_parents(
                (uint32_t*)terminate_flag, parent_list, result_indices,
                itopk_size, search_width
            );
        }
        __syncthreads();


        // --- Step C: 检查终止 ---
        if (*terminate_flag == 1) break;

        // --- Step D: 扩展 ---
        cagra::device::compute_distance_to_child_nodes(
            result_indices + itopk_size,
            result_dists + itopk_size,
            query_buffer, dataset_ptr, knn_graph, graph_degree, dim,
            visited_hash, hash_bitlen, parent_list, search_width
        );
        __syncthreads();
    }

    // -------------------------------------------------------------
    // 5. 结果写回
    // -------------------------------------------------------------
    // 最后再排一次序
    if (tid < 32) {
        if (queue_capacity == 64) load_sort_store<2>(result_dists, result_indices, 64);
        else if (queue_capacity == 128) load_sort_store<4>(result_dists, result_indices, 128);
        else if (queue_capacity == 256) load_sort_store<8>(result_dists, result_indices, 256);
        else if (queue_capacity == 512) load_sort_store<16>(result_dists, result_indices, 512);
        else if (queue_capacity == 1024) load_sort_store<32>(result_dists, result_indices, 1024);
        else if (queue_capacity == 2048) load_sort_store<64>(result_dists, result_indices, 2048);
        else if (queue_capacity == 4096) load_sort_store<128>(result_dists, result_indices, 4096);
        else if (queue_capacity == 8192) load_sort_store<256>(result_dists, result_indices, 8192);
    }
    __syncthreads();

    uint32_t output_offset = query_id * topk;
    
    for (uint32_t i = tid; i < topk; i += blockDim.x) {
        uint32_t idx = result_indices[i] & 0x7FFFFFFF;
        float dist = result_dists[i];
        if (result_indices_ptr) result_indices_ptr[output_offset + i] = idx;
        if (result_distances_ptr) result_distances_ptr[output_offset + i] = dist;
    }


    // if (tid == 0) printf("[iter] iter nums is %d\n", iter);
    if (tid == 0 && num_executed_iterations) {
        num_executed_iterations[query_id] = iter;
    }
}

} // namespace device
} // namespace cagra