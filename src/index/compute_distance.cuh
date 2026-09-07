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

__device__ __forceinline__ float calc_l2_dist_96(const float* vec_a, const float* vec_b) {
    const int lane_id = threadIdx.x % 32;
    float sum_sq = 0.0f;

    // 96 维: 每个线程处理 3 个元素 (32 * 3 = 96)
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
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

// Compile-time dimension/team-size variant used by the specialized range kernel.
// A team is a power-of-two subgroup contained in one warp.
template <uint32_t Dim, uint32_t TeamSize>
__device__ __forceinline__ float calc_l2_dist_team(const float* vec_a, const float* vec_b) {
    static_assert(TeamSize == 2 || TeamSize == 4 || TeamSize == 8 || TeamSize == 16 || TeamSize == 32,
                  "TeamSize must be 2, 4, 8, 16, or 32");
    static_assert(Dim % TeamSize == 0, "Dim must be divisible by TeamSize");

    constexpr uint32_t kTeamMask = TeamSize == 32 ? 0xffffffffu : ((1u << TeamSize) - 1u);
    const uint32_t warp_lane = threadIdx.x & 31u;
    const uint32_t team_lane = warp_lane & (TeamSize - 1u);
    const uint32_t team_base = warp_lane & ~(TeamSize - 1u);
    const uint32_t team_mask = kTeamMask << team_base;

    float sum_sq = 0.0f;
    #pragma unroll
    for (uint32_t i = team_lane; i < Dim; i += TeamSize) {
        const float diff = vec_a[i] - vec_b[i];
        sum_sq += diff * diff;
    }

    #pragma unroll
    for (uint32_t offset = TeamSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(team_mask, sum_sq, offset, TeamSize);
    }
    return sum_sq;
}

template <uint32_t Dim, uint32_t TeamSize>
__device__ inline void compute_distance_to_init_nodes_specialized(
    uint32_t* result_indices,
    float* result_distances,
    const float* query_buffer,
    const float* dataset_ptr,
    size_t num_dataset,
    uint32_t result_buffer_size,
    uint32_t target_num_seeds,
    const uint32_t* seed_ptr,
    uint32_t num_provided_seeds,
    uint64_t rand_xor_mask,
    uint32_t* visited_hash,
    uint32_t hash_bitlen,
    uint32_t* visited_count = nullptr)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t team_lane = tid & (TeamSize - 1u);
    const uint32_t team_id = tid / TeamSize;
    const uint32_t num_teams = blockDim.x / TeamSize;

    for (uint32_t i = tid; i < result_buffer_size; i += blockDim.x) {
        result_indices[i] = 0xFFFFFFFF;
        result_distances[i] = 3.40282e38f;
    }
    __syncthreads();

    for (uint32_t i = team_id; i < target_num_seeds; i += num_teams) {
        uint32_t node_id = 0xFFFFFFFF;
        if (seed_ptr != nullptr && i < num_provided_seeds) node_id = seed_ptr[i];
        if (node_id >= num_dataset) node_id = (rand_xor_mask * (i + 1)) % num_dataset;

        const float* node_ptr = dataset_ptr + static_cast<size_t>(node_id) * Dim;
        const float dist = calc_l2_dist_team<Dim, TeamSize>(query_buffer, node_ptr);
        if (team_lane == 0) {
            result_indices[i] = node_id;
            result_distances[i] = dist;
            if (cagra::hashmap::insert(visited_hash, hash_bitlen, node_id) &&
                visited_count != nullptr) {
                atomicAdd(visited_count, 1u);
            }
        }
    }
    __syncthreads();
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
    uint32_t hash_bitlen,
    uint32_t* visited_count = nullptr
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
        else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_buffer, node_ptr);
        else {
            // 对于非特殊维度，调用通用版本
            printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
        }

        // 写入结果队列的前 num_seeds 个位置
        if (lane_id == 0) {
            result_indices[i] = node_id;
            result_distances[i] = dist;
            
            // 别忘了加入 Hashmap，防止重复访问
            if (cagra::hashmap::insert(visited_hash, hash_bitlen, node_id) && visited_count != nullptr) {
                atomicAdd(visited_count, 1u);
            }
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
    uint32_t hash_bitlen,
    uint32_t* visited_count = nullptr
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
        else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_buffer, node_ptr);
        else {
            // 对于非特殊维度，调用通用版本
            printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
        }

        // 4. 写入队列 & 哈希表 (仅 Lane 0 执行)
        if (lane_id == 0) {
            // 写入结果队列
            result_indices[i] = node_id;
            result_distances[i] = dist;
            
            if (cagra::hashmap::insert(visited_hash, hash_bitlen, node_id) && visited_count != nullptr) {
                atomicAdd(visited_count, 1u);
            }
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
    uint32_t search_width,          // 父节点数量
    uint32_t* visited_count = nullptr
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
                    if (not_visited && visited_count != nullptr) atomicAdd(visited_count, 1u);
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
                    else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_buffer, node_ptr);
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
    uint32_t quen_capacity = 0,
    uint32_t* visited_count = nullptr
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
                    if (not_visited && visited_count != nullptr) atomicAdd(visited_count, 1u);
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
                    else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_buffer, node_ptr);
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
    uint64_t* d_ts,                 // 反查表
    uint32_t* visited_count = nullptr,
    unsigned long long* child_profile = nullptr
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
    unsigned long long clk_graph = 0;
    unsigned long long clk_hash = 0;
    unsigned long long clk_filter = 0;
    unsigned long long clk_dist = 0;
    unsigned long long clk_write = 0;
    unsigned long long cnt_in_range = 0;
    unsigned long long cnt_dist = 0;

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
            unsigned long long t_child = 0;
            if (child_profile != nullptr && lane_id == 0) t_child = clock64();
            uint32_t neighbor_id = knn_graph[(size_t)parent_id * graph_degree + neighbor_offset];
            if (child_profile != nullptr && lane_id == 0) clk_graph += clock64() - t_child;

            // 4. 检查邻居是否有效 (填充值)
            if (neighbor_id != 0xFFFFFFFF) {

                // 5. 先做范围过滤，再写 visited hash。
                // Range search 下如果先插入 hash，out-of-range 邻居会快速填满小 hash 表，
                // 让后续 atomicCAS 线性探测退化。
                if (child_profile != nullptr && lane_id == 0) t_child = clock64();
                uint64_t bucket_id = d_ts == nullptr
                    ? static_cast<uint64_t>(neighbor_id)
                    : __ldg(&d_ts[neighbor_id]);
                if (child_profile != nullptr && lane_id == 0) clk_filter += clock64() - t_child;
                if (bucket_id < start_bucket || bucket_id >= end_bucket) {
                    if (lane_id == 0) {
                        candidate_indices[task_id] = 0xFFFFFFFF;
                        candidate_distances[task_id] = 3.40282e38f;
                    }
                    continue;
                }
                if (child_profile != nullptr && lane_id == 0) cnt_in_range++;

                // 6. 查重：Hashmap
                // 只有 Lane 0 负责查重 (原子操作)，结果广播给全 Warp
                // insert 返回 true 表示插入成功(未访问过)，false 表示已存在
                int not_visited = 0;
                if (lane_id == 0) {
                    if (child_profile != nullptr) t_child = clock64();
                    not_visited = cagra::hashmap::insert(visited_hash, hash_bitlen, neighbor_id);
                    if (not_visited && visited_count != nullptr) atomicAdd(visited_count, 1u);
                    if (child_profile != nullptr) clk_hash += clock64() - t_child;
                }
                // 广播查重结果
                not_visited = __shfl_sync(0xFFFFFFFF, not_visited, 0);

                if (not_visited) {
                    // 7. 没访问过 -> 计算距离
                    const float* node_ptr = dataset_ptr + (size_t)neighbor_id * dim;
                    float dist = 3.40282e38f; // MAX_FLOAT
                    if (child_profile != nullptr && lane_id == 0) t_child = clock64();
                    if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
                    else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
                    else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
                    else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
                    else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
                    else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_buffer, node_ptr);
                    else {
                        // 对于非特殊维度，调用通用版本
                        printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
                    }
                    if (child_profile != nullptr && lane_id == 0) {
                        clk_dist += clock64() - t_child;
                        cnt_dist++;
                    }

                    // 7. 写入结果
                    if (lane_id == 0) {
                        if (child_profile != nullptr) t_child = clock64();
                        candidate_indices[task_id] = neighbor_id;
                        candidate_distances[task_id] = dist;
                        if (child_profile != nullptr) clk_write += clock64() - t_child;
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
    if (child_profile != nullptr) {
        unsigned long long graph = 0;
        unsigned long long hash = 0;
        unsigned long long filter = 0;
        unsigned long long dist = 0;
        unsigned long long write = 0;
        unsigned long long in_range = 0;
        unsigned long long dist_count = 0;
        if (lane_id == 0) {
            graph = clk_graph;
            hash = clk_hash;
            filter = clk_filter;
            dist = clk_dist;
            write = clk_write;
            in_range = cnt_in_range;
            dist_count = cnt_dist;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            graph += __shfl_down_sync(0xffffffff, graph, offset);
            hash += __shfl_down_sync(0xffffffff, hash, offset);
            filter += __shfl_down_sync(0xffffffff, filter, offset);
            dist += __shfl_down_sync(0xffffffff, dist, offset);
            write += __shfl_down_sync(0xffffffff, write, offset);
            in_range += __shfl_down_sync(0xffffffff, in_range, offset);
            dist_count += __shfl_down_sync(0xffffffff, dist_count, offset);
        }
        if (lane_id == 0) {
            atomicAdd(child_profile + 0, graph);
            atomicAdd(child_profile + 1, hash);
            atomicAdd(child_profile + 2, filter);
            atomicAdd(child_profile + 3, dist);
            atomicAdd(child_profile + 4, write);
            atomicAdd(child_profile + 5, in_range);
            atomicAdd(child_profile + 6, dist_count);
        }
    }
    // 所有 Warp 完成计算后同步
    __syncthreads();
}

// Compile-time dimension/team-size range expansion. TeamSize=8 allows four
// independent low-dimensional candidates to be processed by each warp.
template <uint32_t Dim, uint32_t TeamSize>
__device__ inline void compute_distance_to_child_nodes_range_specialized(
    uint32_t* candidate_indices,
    float* candidate_distances,
    const float* query_buffer,
    const float* dataset_ptr,
    const uint32_t* knn_graph,
    uint32_t graph_stride,
    uint32_t active_degree,
    uint32_t* visited_hash,
    uint32_t hash_bitlen,
    const uint32_t* parent_list,
    uint32_t search_width,
    uint64_t start_bucket,
    uint64_t end_bucket,
    uint64_t* d_ts,
    uint32_t* visited_count = nullptr,
    unsigned long long* child_profile = nullptr)
{
    static_assert(TeamSize == 2 || TeamSize == 4 || TeamSize == 8 || TeamSize == 16 || TeamSize == 32,
                  "TeamSize must be 2, 4, 8, 16, or 32");
    (void)active_degree;

    constexpr uint32_t kTeamMask = TeamSize == 32 ? 0xffffffffu : ((1u << TeamSize) - 1u);
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_lane = tid & 31u;
    const uint32_t team_lane = warp_lane & (TeamSize - 1u);
    const uint32_t team_base = warp_lane & ~(TeamSize - 1u);
    const uint32_t team_mask = kTeamMask << team_base;
    const uint32_t team_id = tid / TeamSize;
    const uint32_t num_teams = blockDim.x / TeamSize;

    const uint32_t total_tasks = search_width * graph_stride;
    unsigned long long clk_graph = 0;
    unsigned long long clk_hash = 0;
    unsigned long long clk_filter = 0;
    unsigned long long clk_dist = 0;
    unsigned long long clk_write = 0;
    unsigned long long cnt_in_range = 0;
    unsigned long long cnt_dist = 0;

    for (uint32_t task_id = team_id; task_id < total_tasks; task_id += num_teams) {
        const uint32_t parent_idx = task_id / graph_stride;
        const uint32_t neighbor_offset = task_id % graph_stride;
        const uint32_t parent_id = parent_list[parent_idx];

        if (parent_id == 0xFFFFFFFF) {
            if (team_lane == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
            continue;
        }

        unsigned long long t_child = 0;
        if (child_profile != nullptr && team_lane == 0) t_child = clock64();
        uint32_t neighbor_id = 0xFFFFFFFF;
        if (team_lane == 0) {
            neighbor_id = knn_graph[static_cast<size_t>(parent_id) * graph_stride + neighbor_offset];
        }
        neighbor_id = __shfl_sync(team_mask, neighbor_id, 0, TeamSize);
        if (child_profile != nullptr && team_lane == 0) clk_graph += clock64() - t_child;

        if (neighbor_id == 0xFFFFFFFF) {
            if (team_lane == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
            continue;
        }

        if (child_profile != nullptr && team_lane == 0) t_child = clock64();
        int in_range = 0;
        if (team_lane == 0) {
            const uint64_t bucket_id = d_ts == nullptr
                ? static_cast<uint64_t>(neighbor_id)
                : __ldg(&d_ts[neighbor_id]);
            in_range = bucket_id >= start_bucket && bucket_id < end_bucket;
        }
        in_range = __shfl_sync(team_mask, in_range, 0, TeamSize);
        if (child_profile != nullptr && team_lane == 0) clk_filter += clock64() - t_child;
        if (!in_range) {
            if (team_lane == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
            continue;
        }
        if (child_profile != nullptr && team_lane == 0) cnt_in_range++;

        int not_visited = 0;
        if (team_lane == 0) {
            if (child_profile != nullptr) t_child = clock64();
            not_visited = cagra::hashmap::insert(visited_hash, hash_bitlen, neighbor_id);
            if (not_visited && visited_count != nullptr) atomicAdd(visited_count, 1u);
            if (child_profile != nullptr) clk_hash += clock64() - t_child;
        }
        not_visited = __shfl_sync(team_mask, not_visited, 0, TeamSize);

        if (!not_visited) {
            if (team_lane == 0) {
                candidate_indices[task_id] = 0xFFFFFFFF;
                candidate_distances[task_id] = 3.40282e38f;
            }
            continue;
        }

        if (child_profile != nullptr && team_lane == 0) t_child = clock64();
        const float* node_ptr = dataset_ptr + static_cast<size_t>(neighbor_id) * Dim;
        const float dist = calc_l2_dist_team<Dim, TeamSize>(query_buffer, node_ptr);
        if (child_profile != nullptr && team_lane == 0) {
            clk_dist += clock64() - t_child;
            cnt_dist++;
        }

        if (team_lane == 0) {
            if (child_profile != nullptr) t_child = clock64();
            candidate_indices[task_id] = neighbor_id;
            candidate_distances[task_id] = dist;
            if (child_profile != nullptr) clk_write += clock64() - t_child;
        }
    }

    if (child_profile != nullptr) {
        unsigned long long graph = team_lane == 0 ? clk_graph : 0;
        unsigned long long hash = team_lane == 0 ? clk_hash : 0;
        unsigned long long filter = team_lane == 0 ? clk_filter : 0;
        unsigned long long dist = team_lane == 0 ? clk_dist : 0;
        unsigned long long write = team_lane == 0 ? clk_write : 0;
        unsigned long long in_range = team_lane == 0 ? cnt_in_range : 0;
        unsigned long long dist_count = team_lane == 0 ? cnt_dist : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            graph += __shfl_down_sync(0xffffffff, graph, offset);
            hash += __shfl_down_sync(0xffffffff, hash, offset);
            filter += __shfl_down_sync(0xffffffff, filter, offset);
            dist += __shfl_down_sync(0xffffffff, dist, offset);
            write += __shfl_down_sync(0xffffffff, write, offset);
            in_range += __shfl_down_sync(0xffffffff, in_range, offset);
            dist_count += __shfl_down_sync(0xffffffff, dist_count, offset);
        }
        if (warp_lane == 0) {
            atomicAdd(child_profile + 0, graph);
            atomicAdd(child_profile + 1, hash);
            atomicAdd(child_profile + 2, filter);
            atomicAdd(child_profile + 3, dist);
            atomicAdd(child_profile + 4, write);
            atomicAdd(child_profile + 5, in_range);
            atomicAdd(child_profile + 6, dist_count);
        }
    }
    __syncthreads();
}

} // namespace device
} // namespace cagra
