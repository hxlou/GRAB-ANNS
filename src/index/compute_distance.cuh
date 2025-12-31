#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "cagra.cuh"
#include "hashmap.cuh"

namespace cagra {
namespace device {

// ============================================================================
// 基础算子：Warp 级 L2 距离计算 (针对 1024 维优化)
// ============================================================================
// 假设调用此函数的 32 个线程是一个 Warp
__device__ __forceinline__ float calc_l2_dist_1024(const float* vec_a, const float* vec_b) {
    const int lane_id = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 每个线程处理 32 个元素 (32 * 32 = 1024)
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int idx = i * 32 + lane_id;
        float diff = vec_a[idx] - vec_b[idx];
        sum_sq += diff * diff;
    }

    // Warp 归约求和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    
    return sum_sq;
}

__device__ __forceinline__ float calc_l2_dist_2048(const float* vec_a, const float* vec_b) {
    const int lane_id = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 每个线程处理 32 个元素 (32 * 64 = 2048)
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        int idx = i * 32 + lane_id;
        float diff = vec_a[idx] - vec_b[idx];
        sum_sq += diff * diff;
    }

    // Warp 归约求和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    
    return sum_sq;
}


__device__ __forceinline__ float calc_l2_dist_960(const float* vec_a, const float* vec_b) {
    const int lane_id = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 每个线程处理 32 个元素 (32 * 32 = 1024)
    #pragma unroll
    for (int i = 0; i < 30; ++i) {
        int idx = i * 32 + lane_id;
        float diff = vec_a[idx] - vec_b[idx];
        sum_sq += diff * diff;
    }

    // Warp 归约求和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    
    return sum_sq;
}

__device__ __forceinline__ float calc_l2_dist_256(const float* vec_a, const float* vec_b) {
    const int lane_id = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 每个线程处理 32 个元素 (32 * 32 = 1024)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane_id;
        float diff = vec_a[idx] - vec_b[idx];
        sum_sq += diff * diff;
    }

    // Warp 归约求和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    
    return sum_sq;
}

__device__ __forceinline__ float calc_l2_dist_128(const float* vec_a, const float* vec_b) {
    const int lane_id = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 每个线程处理 32 个元素 (32 * 32 = 1024)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = i * 32 + lane_id;
        float diff = vec_a[idx] - vec_b[idx];
        sum_sq += diff * diff;
    }

    // Warp 归约求和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    
    return sum_sq;
}

// ============================================================================
// 阶段 1: 初始化 (随机选取节点计算距离)
// ============================================================================
__device__ inline void compute_distance_to_random_nodes(
    uint32_t* result_indices,       // [Output] 结果索引队列
    float* result_distances,        // [Output] 结果距离队列
    const float* query_buffer,      // Shared Mem 中的 Query
    const float* dataset_ptr,       // Global Mem 数据集
    size_t num_dataset,             // 数据集大小
    uint32_t dim,                   // 维度 (1024)
    uint32_t result_buffer_size,    // 队列总容量
    uint32_t num_seeds,             // 需要生成的随机种子数量
    uint64_t rand_xor_mask,         // 随机数掩码
    uint32_t* visited_hash,         // Hashmap
    uint32_t hash_bitlen
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    // 假设 BlockDim = 256, 也就是有 8 个 Warp
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32;

    // 1. 并行初始化整个结果队列为 INVALID / MAX_FLOAT
    for (uint32_t i = tid; i < result_buffer_size; i += blockDim.x) {
        result_indices[i] = 0xFFFFFFFF;
        result_distances[i] = 3.40282e38f; // FLT_MAX
    }
    __syncthreads(); // 必须同步，确保初始化完成

    // 2. 每个 Warp 负责计算若干个种子的距离
    // 我们需要填充 num_seeds 个种子
    for (uint32_t i = warp_id; i < num_seeds; i += num_warps) {
        
        // 简单的伪随机数生成 ID (对应原算法的 num_random_samplings)
        // 简单的线性同余或者异或哈希
        uint32_t node_id = (rand_xor_mask * (i + 1)) % num_dataset;

        // 计算距离
        const float* node_ptr = dataset_ptr + (size_t)node_id * dim;
        float dist = 3.40282e38f; // MAX_FLOAT
        if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
        else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
        else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
        else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
        else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
        else {
            // 对于非特殊维度，调用通用版本
            printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
        }

        // 写入结果队列的前 num_seeds 个位置
        if (lane_id == 0) {
            result_indices[i] = node_id;
            result_distances[i] = dist;
            
            // 别忘了加入 Hashmap，防止重复访问
            cagra::hashmap::insert(visited_hash, hash_bitlen, node_id);
        }
    }
    __syncthreads();
}

__device__ inline void compute_distance_to_init_nodes(
    uint32_t* result_indices,       // [Output]
    float* result_distances,        // [Output]
    const float* query_buffer,      // Shared Mem Query
    const float* dataset_ptr,       // Global Dataset
    size_t num_dataset,             // N
    uint32_t dim,                   // 1024
    uint32_t result_buffer_size,    // 队列容量
    uint32_t target_num_seeds,      // 目标需要生成的种子总数 (即 num_seeds)
    const uint32_t* seed_ptr,       // [Input] 外部提供的种子列表 (可以是 nullptr)
    uint32_t num_provided_seeds,    // [Input] 外部提供的种子数量
    uint64_t rand_xor_mask,         // 随机掩码
    uint32_t* visited_hash,         // Hashmap
    uint32_t hash_bitlen
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32;

    // 1. 并行初始化结果队列
    for (uint32_t i = tid; i < result_buffer_size; i += blockDim.x) {
        result_indices[i] = 0xFFFFFFFF;
        result_distances[i] = 3.40282e38f; // FLT_MAX
    }
    __syncthreads();

    // 2. 每个 Warp 负责填充一部分种子
    // 循环直到填满 target_num_seeds 个位置
    for (uint32_t i = warp_id; i < target_num_seeds; i += num_warps) {
        
        uint32_t node_id = 0xFFFFFFFF;

        if (seed_ptr != nullptr && i < num_provided_seeds) {
            node_id = seed_ptr[i];
        }

        if (node_id >= num_dataset) {
            node_id = (rand_xor_mask * (i + 1)) % num_dataset;
        }

        // 3. 计算距离 (Warp 级并行)
        const float* node_ptr = dataset_ptr + (size_t)node_id * dim;
        float dist = 3.40282e38f; // MAX_FLOAT
        if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
        else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
        else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
        else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
        else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
        else {
            // 对于非特殊维度，调用通用版本
            printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
        }

        // 4. 写入队列 & 哈希表 (仅 Lane 0 执行)
        if (lane_id == 0) {
            // 写入结果队列
            result_indices[i] = node_id;
            result_distances[i] = dist;
            
            cagra::hashmap::insert(visited_hash, hash_bitlen, node_id);
        }
    }
    __syncthreads();
}

// ============================================================================
// 阶段 2: 扩展 (计算子节点距离)
// ============================================================================
__device__ inline void compute_distance_to_child_nodes(
    uint32_t* candidate_indices,    // [Output] 写入这里 (接在 itopk 后面)
    float* candidate_distances,     // [Output] 写入这里
    const float* query_buffer,      // Query
    const float* dataset_ptr,       // Dataset
    const uint32_t* knn_graph,      // Graph
    uint32_t graph_degree,          // 图度数 (32/64)
    uint32_t dim,                   // 1024
    uint32_t* visited_hash,         // Hashmap
    uint32_t hash_bitlen,
    const uint32_t* parent_list,    // [Input] 父节点列表
    uint32_t search_width           // 父节点数量
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32; // 8

    // 任务总量：search_width 个父节点 * graph_degree 个邻居
    // 我们将其展平，分配给各个 Warp
    // 每个 Warp 负责处理一个 (parent, neighbor) 对
    
    // 注意：这里的内存写入位置是 candidate_indices[k]
    // k 范围是 [0, search_width * graph_degree - 1]
    
    // 外层循环：遍历所有父节点
    // 为了简化逻辑，我们让每个 Warp 负责处理 "一个父节点的一组邻居"
    // 或者更细粒度：所有 Warp 共同瓜分 "所有父节点的所有邻居"
    
    // 采用更细粒度的策略：
    uint32_t total_tasks = search_width * graph_degree;

    for (uint32_t task_id = warp_id; task_id < total_tasks; task_id += num_warps) {
        
        // 1. 解码任务：当前处理第几个父节点的第几个邻居？
        uint32_t parent_idx = task_id / graph_degree;
        uint32_t neighbor_offset = task_id % graph_degree;

        // 2. 获取父节点 ID
        uint32_t parent_id = parent_list[parent_idx];
        
        // 检查父节点是否有效
        if (parent_id != 0xFFFFFFFF) {
            
            // 3. 查图：获取邻居 ID
            // knn_graph 是 [N, degree] 的行主序
            // 邻居位置 = parent_id * degree + offset
            uint32_t neighbor_id = knn_graph[(size_t)parent_id * graph_degree + neighbor_offset];

            // 4. 检查邻居是否有效 (填充值)
            if (neighbor_id != 0xFFFFFFFF) {
                
                // 5. 查重：Hashmap
                // 只有 Lane 0 负责查重 (原子操作)，结果广播给全 Warp
                // insert 返回 true 表示插入成功(未访问过)，false 表示已存在
                int not_visited = 0;
                if (lane_id == 0) {
                    not_visited = cagra::hashmap::insert(visited_hash, hash_bitlen, neighbor_id);
                }
                // 广播查重结果
                not_visited = __shfl_sync(0xFFFFFFFF, not_visited, 0);

                if (not_visited) {
                    // 6. 没访问过 -> 计算距离
                    const float* node_ptr = dataset_ptr + (size_t)neighbor_id * dim;
                    float dist = 3.40282e38f; // MAX_FLOAT
                    if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
                    else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
                    else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
                    else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
                    else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
                    else {
                        // 对于非特殊维度，调用通用版本
                        printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
                    }

                    // 7. 写入结果
                    if (lane_id == 0) {
                        candidate_indices[task_id] = neighbor_id;
                        candidate_distances[task_id] = dist;
                    }
                } else {
                    // 已访问过 -> 写入无效值
                    if (lane_id == 0) {
                        candidate_indices[task_id] = 0xFFFFFFFF;
                        candidate_distances[task_id] = 3.40282e38f;
                    }
                }
            } else {
                // 无效邻居 -> 写入无效值
                if (lane_id == 0) {
                    candidate_indices[task_id] = 0xFFFFFFFF;
                    candidate_distances[task_id] = 3.40282e38f;
                }
            }
        } else {
            // 无效父节点 -> 写入无效值
            if (lane_id == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
        }
    }
    // 所有 Warp 完成计算后同步
    __syncthreads();
}

// ============================================================================
// 阶段 2 (变体): 局部扩展 (Local-Only Expansion)
// 适用于桶内搜索，强制只访问前 active_degree 个邻居 (Local Edges)
// ============================================================================
__device__ inline void compute_distance_to_child_nodes_strided(
    uint32_t* candidate_indices,    
    float* candidate_distances,     
    const float* query_buffer,      
    const float* dataset_ptr,       
    const uint32_t* knn_graph,      
    uint32_t graph_stride,          // 物理宽度 (32)
    uint32_t active_degree,         // 逻辑宽度 (28)
    uint32_t dim,                   
    uint32_t* visited_hash,         
    uint32_t hash_bitlen,
    const uint32_t* parent_list,    
    uint32_t search_width,
    uint32_t* itopk_indices = nullptr,
    uint32_t quen_capacity = 0   
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32; // 8

    // 任务总量：search_width 个父节点 * graph_degree 个邻居
    // 我们将其展平，分配给各个 Warp
    // 每个 Warp 负责处理一个 (parent, neighbor) 对
    
    // 注意：这里的内存写入位置是 candidate_indices[k]
    // k 范围是 [0, search_width * graph_degree - 1]
    
    // 外层循环：遍历所有父节点
    // 为了简化逻辑，我们让每个 Warp 负责处理 "一个父节点的一组邻居"
    // 或者更细粒度：所有 Warp 共同瓜分 "所有父节点的所有邻居"
    
    // 采用更细粒度的策略：
    uint32_t graph_degree = graph_stride; // 物理宽度
    uint32_t total_tasks = search_width * graph_degree;

    for (uint32_t task_id = warp_id; task_id < total_tasks; task_id += num_warps) {
        
        // 1. 解码任务：当前处理第几个父节点的第几个邻居？
        uint32_t parent_idx = task_id / graph_degree;
        uint32_t neighbor_offset = task_id % graph_degree;

        // 仅处理前 active_degree 个邻居
        if (neighbor_offset >= active_degree) {
            // 写入无效值
            if (lane_id == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
            continue;
        }

        // 2. 获取父节点 ID
        uint32_t parent_id = parent_list[parent_idx];
        // 检查父节点是否有效
        if (parent_id != 0xFFFFFFFF) {
            
            // 3. 查图：获取邻居 ID
            // knn_graph 是 [N, degree] 的行主序
            // 邻居位置 = parent_id * degree + offset
            uint32_t neighbor_id = knn_graph[(size_t)parent_id * graph_degree + neighbor_offset];

            // 4. 检查邻居是否有效 (填充值)
            if (neighbor_id != 0xFFFFFFFF) {
                
                // 5. 查重：Hashmap
                // 只有 Lane 0 负责查重 (原子操作)，结果广播给全 Warp
                // insert 返回 true 表示插入成功(未访问过)，false 表示已存在
                int not_visited = 0;
                if (lane_id == 0) {
                    not_visited = cagra::hashmap::insert(visited_hash, hash_bitlen, neighbor_id);
                }
                // 广播查重结果
                not_visited = __shfl_sync(0xFFFFFFFF, not_visited, 0);

                if (not_visited) {
                    // 6. 没访问过 -> 计算距离
                    const float* node_ptr = dataset_ptr + (size_t)neighbor_id * dim;
                    float dist = 3.40282e38f; // MAX_FLOAT
                    if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
                    else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
                    else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
                    else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
                    else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
                    else {
                        // 对于非特殊维度，调用通用版本
                        printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
                    }

                    // 7. 写入结果
                    if (lane_id == 0) {
                        candidate_indices[task_id] = neighbor_id;
                        candidate_distances[task_id] = dist;
                    }
                } else {
                    // 已访问过 -> 写入无效值
                    if (lane_id == 0) {
                        candidate_indices[task_id] = 0xFFFFFFFF;
                        candidate_distances[task_id] = 3.40282e38f;
                    }
                }
            } else {
                // 无效邻居 -> 写入无效值
                if (lane_id == 0) {
                    candidate_indices[task_id] = 0xFFFFFFFF;
                    candidate_distances[task_id] = 3.40282e38f;
                }
            }
        } else {
            // 无效父节点 -> 写入无效值
            if (lane_id == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
        }
    }
    // 所有 Warp 完成计算后同步
    __syncthreads();
}


// ============================================================================
// 阶段 3 (变体): 局部扩展 (Local-Only Expansion)
// 给定搜索桶的范围，只访问这些桶内的邻居节点
// ============================================================================
__device__ inline void compute_distance_to_child_nodes_range(
    uint32_t* candidate_indices,    
    float* candidate_distances,     
    const float* query_buffer,      
    const float* dataset_ptr,       
    const uint32_t* knn_graph,      
    uint32_t graph_stride,          // 物理宽度 (32)
    uint32_t active_degree,         // 逻辑宽度 (28)
    uint32_t dim,         
    uint32_t* visited_hash,
    uint32_t hash_bitlen,
    const uint32_t* parent_list,    
    uint32_t search_width,
    uint64_t start_bucket,          // [start_bucket, end_bucket)
    uint64_t end_bucket,
    uint64_t* d_ts                  // 反查表
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32; // 8

    // 任务总量：search_width 个父节点 * graph_degree 个邻居
    // 我们将其展平，分配给各个 Warp
    // 每个 Warp 负责处理一个 (parent, neighbor) 对
    
    // 注意：这里的内存写入位置是 candidate_indices[k]
    // k 范围是 [0, search_width * graph_degree - 1]
    
    // 外层循环：遍历所有父节点
    // 为了简化逻辑，我们让每个 Warp 负责处理 "一个父节点的一组邻居"
    // 或者更细粒度：所有 Warp 共同瓜分 "所有父节点的所有邻居"
    
    // 采用更细粒度的策略：
    uint32_t graph_degree = graph_stride; // 物理宽度
    uint32_t total_tasks = search_width * graph_degree;

    for (uint32_t task_id = warp_id; task_id < total_tasks; task_id += num_warps) {
        
        // 1. 解码任务：当前处理第几个父节点的第几个邻居？
        uint32_t parent_idx = task_id / graph_degree;
        uint32_t neighbor_offset = task_id % graph_degree;

        // 2. 获取父节点 ID
        uint32_t parent_id = parent_list[parent_idx];
        
        // 检查父节点是否有效
        if (parent_id != 0xFFFFFFFF) {
            
            // 3. 查图：获取邻居 ID
            // knn_graph 是 [N, degree] 的行主序
            // 邻居位置 = parent_id * degree + offset
            uint32_t neighbor_id = knn_graph[(size_t)parent_id * graph_degree + neighbor_offset];

            // 4. 检查邻居是否有效 (填充值)
            if (neighbor_id != 0xFFFFFFFF) {

                // 5. 查重：Hashmap
                // 只有 Lane 0 负责查重 (原子操作)，结果广播给全 Warp
                // insert 返回 true 表示插入成功(未访问过)，false 表示已存在
                int not_visited = 0;
                if (lane_id == 0) {
                    not_visited = cagra::hashmap::insert(visited_hash, hash_bitlen, neighbor_id);
                }
                // 广播查重结果
                not_visited = __shfl_sync(0xFFFFFFFF, not_visited, 0);

                // 检查邻居是否在指定桶范围内 优先哈希表过滤
                uint64_t bucket_id = __ldg(&d_ts[neighbor_id]);
                if (bucket_id < start_bucket || bucket_id >= end_bucket) {
                    // 不在范围内，写入无效值
                    continue;
                }

                if (not_visited) {
                    // 6. 没访问过 -> 计算距离
                    const float* node_ptr = dataset_ptr + (size_t)neighbor_id * dim;
                    float dist = 3.40282e38f; // MAX_FLOAT
                    if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
                    else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
                    else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
                    else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
                    else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
                    else {
                        // 对于非特殊维度，调用通用版本
                        printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
                    }

                    // 7. 写入结果
                    if (lane_id == 0) {
                        candidate_indices[task_id] = neighbor_id;
                        candidate_distances[task_id] = dist;
                    }
                } else {
                    // 已访问过 -> 写入无效值
                }
            }
        }
    }
    // 所有 Warp 完成计算后同步
    __syncthreads();
}

} // namespace device
} // namespace cagra