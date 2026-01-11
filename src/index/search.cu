#include "search.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include "cagra.cuh"
#include "config.cuh"
#include "hashmap.cuh"
#include "bitonic.cuh"
#include "compute_distance.cuh"
#include "warp_merge_sort.cuh"
#include "radix_sort.cuh"
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
    uint32_t* pre_hashmap,
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
    uint32_t* visited_hash = nullptr;
    if (hash_bitlen < 14) {
        visited_hash = (uint32_t*)(smem + offset);
        offset += ((1u << hash_bitlen) * sizeof(uint32_t) + 15) & ~15;
    } else {
        // 申请一个存放在显存上的hash map
        visited_hash = pre_hashmap + (blockIdx.x * (1u << hash_bitlen));
    }

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
    uint32_t hash_reset_iter = 30; // 每隔这么多轮重置 Hashmap

    uint64_t step1 = 0;
    uint64_t step2 = 0;
    uint64_t step3 = 0;

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
        auto t1 = clock64();      
        if (queue_capacity == 64 && tid < 32)       load_sort_store<2>(result_dists, result_indices, 64);
        else if (queue_capacity == 128 && tid < 32) load_sort_store<4>(result_dists, result_indices, 128);
        else if (queue_capacity == 256 && tid < 32) cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
        else if (queue_capacity == 512 && tid < 32) cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
        else if (queue_capacity == 32 * 32) cagra::radix::load_sort_store(result_dists, result_indices, 1024);
        else {
            // 不支持的容量大小
            if (tid == 0) {
                printf(">> [search_kernel_range] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
            }
        }
        __syncthreads();
        auto t2 = clock64();

        // --- Step B: 选父节点 ---
        auto t3 = clock64();
        if (tid < 32) {
            cagra::device::pickup_next_parents(
                (uint32_t*)terminate_flag, parent_list, result_indices,
                itopk_size, search_width
            );
        }
        __syncthreads();
        auto t4 = clock64();

        // --- Step C: 检查终止 ---
        if (*terminate_flag == 1) break;

        // --- Step D: 扩展 ---
        auto t5 = clock64();
        cagra::device::compute_distance_to_child_nodes(
            result_indices + itopk_size,
            result_dists + itopk_size,
            query_buffer, dataset_ptr, knn_graph, graph_degree, dim,
            visited_hash, hash_bitlen, parent_list, search_width
        );
        __syncthreads();
        auto t6 = clock64();

        step1 += t2 - t1;
        step2 += t4 - t3;
        step3 += t6 - t5;
    }

    // -------------------------------------------------------------
    // 5. 结果写回
    // -------------------------------------------------------------
    // 最后再排一次序
    if (queue_capacity == 64 && tid < 32)       load_sort_store<2>(result_dists, result_indices, 64);
    else if (queue_capacity == 128 && tid < 32) load_sort_store<4>(result_dists, result_indices, 128);
    else if (queue_capacity == 256 && tid < 32) cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
    else if (queue_capacity == 512 && tid < 32) cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
    else if (queue_capacity == 32 * 32) cagra::radix::load_sort_store(result_dists, result_indices, 1024);
    else {
        // 不支持的容量大小
        if (tid == 0) {
            printf(">> [search_kernel_range] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
        }
    }
    __syncthreads();

    uint32_t output_offset = query_id * topk;
    
    for (uint32_t i = tid; i < topk; i += blockDim.x) {
        uint32_t idx = result_indices[i] & 0x7FFFFFFF;
        float dist = result_dists[i];
        if (result_indices_ptr) result_indices_ptr[output_offset + i] = idx;
        if (result_distances_ptr) result_distances_ptr[output_offset + i] = dist;
    }

    // if (tid == 0 && query_id == 1) {
    //     printf("query %u finished in %u iterations, and queue capacity is %u, and total time cost is %llu.\n", query_id, iter, queue_capacity, step1 + step2 + step3);
    //     printf("sort time: %llu, pickup time: %llu, expand time: %llu\n", step1, step2, step3);
    //     // 转换成百分比再输出一下
    //     printf("sort perc: %.2f%%, pickup perc: %.2f%%, expand perc: %.2f%%\n", 
    //         step1 * 100.0 / (step1 + step2 + step3),
    //         step2 * 100.0 / (step1 + step2 + step3),
    //         step3 * 100.0 / (step1 + step2 + step3)
    //     );
    // }


    // if (tid == 0) printf("[iter] iter nums is %d\n", iter);
}

// =============================================================================
// Bucket Search Kernel: 针对特定桶的局部搜索
// =============================================================================
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
) {
    // 1. Shared Memory Init (完全复用，代码一样)
    extern __shared__ uint8_t smem[]; 
    size_t offset = 0;

    float* query_buffer = (float*)(smem + offset);
    offset += (dim * sizeof(float) + 15) & ~15;

    uint32_t* visited_hash = nullptr;
    if (hash_bitlen < 14) {
        visited_hash = (uint32_t*)(smem + offset);
        offset += ((1u << hash_bitlen) * sizeof(uint32_t) + 15) & ~15;     
    } else {
        visited_hash = pre_hashmap + (blockIdx.x * (1u << hash_bitlen));
    }

    uint32_t* result_indices = (uint32_t*)(smem + offset);
    offset += (queue_capacity * sizeof(uint32_t) + 15) & ~15;

    float* result_dists = (float*)(smem + offset);
    offset += (queue_capacity * sizeof(float) + 15) & ~15;

    uint32_t* parent_list = (uint32_t*)(smem + offset);
    offset += (search_width * sizeof(uint32_t) + 15) & ~15;

    volatile uint32_t* terminate_flag = (uint32_t*)(smem + offset);

    // 2. Thread Init
    const uint32_t query_id = blockIdx.x;
    if (query_id >= num_queries) return;
    const uint32_t tid = threadIdx.x;

    const float* global_query = queries_ptr + (size_t)query_id * dim;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = global_query[i];
    }
    
    if (tid == 0) *terminate_flag = 0;
    cagra::hashmap::init(visited_hash, hash_bitlen);
    __syncthreads(); 

    // 3. 初始种子阶段
    // 【注意】这里我们假设外部一定提供了足够的桶内种子，或者 num_seeds 设为 0
    // 如果没有种子，我们不能全图随机，必须在桶内随机。
    // 但为了简化，我们复用 init_nodes，并假设上层逻辑保证了种子质量。
    // 如果上层没传种子，这里的兜底随机逻辑是全图的，可能会“跳出桶”。
    // *修正策略*：如果你希望 Kernel 内部支持桶内随机，你需要传 bucket_start 和 bucket_size 进来。
    // 但既然你说了“Host 端直接生成”，那我们就信任 seed_ptr。
    
    const uint32_t* local_seed_ptr = nullptr;
    if (seed_ptr != nullptr) {
        local_seed_ptr = seed_ptr + (size_t)query_id * num_provided_seeds;
    }

    cagra::device::compute_distance_to_init_nodes(
        result_indices, result_dists, query_buffer, dataset_ptr,
        num_dataset, dim, queue_capacity, 
        num_seeds,          
        local_seed_ptr,     
        num_provided_seeds, 
        rand_xor_mask, visited_hash, hash_bitlen
    );
    __syncthreads();

    // 4. 主循环
    uint32_t iter = 0;
    uint32_t hash_reset_iter = 30;

    // auto step1 = 0;
    // auto step2 = 0;
    // auto step3 = 0;


    for (; iter < max_iterations; ++iter) {
        if (iter > 0 && (iter % hash_reset_iter == 0)) {
            cagra::hashmap::init(visited_hash, hash_bitlen);
            __syncthreads();
            cagra::hashmap::restore(visited_hash, hash_bitlen, result_indices, itopk_size);
            __syncthreads();
        }

        // auto t1 = clock64();
        // A. Sort (完全复用)
        // if (tid < 32) {
            if (queue_capacity == 64 && tid < 32)       load_sort_store<2>(result_dists, result_indices, 64);
            else if (queue_capacity == 128 && tid < 32) load_sort_store<4>(result_dists, result_indices, 128);
            else if (queue_capacity == 256 && tid < 32) cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
            else if (queue_capacity == 512 && tid < 32) cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
            else if (queue_capacity == 32 * 32) cagra::radix::load_sort_store(result_dists, result_indices, 1024);
            else if (queue_capacity == 32 * 64) cagra::radix::load_sort_store(result_dists, result_indices, 2048);
            // else if (queue_capacity == 32 * 128) cagra::radix::load_sort_store<128>(result_dists, result_indices, 4096);
            // else if (queue_capacity == 32 * 256) cagra::radix::load_sort_store<256>(result_dists, result_indices, 8192);
            else {
                // 不支持的容量大小
                if (tid == 0) {
                    printf(">> [search_kernel_bucket] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
                }
            }
        // }
        __syncthreads();

        // // 输出itopk中的内容，调试用
        // if (tid == 0 && query_id == 0) {
        //     printf("Now is After Iteration %u for Query %u:\n", iter, query_id);
        //     printf("itopk index are as follows: ");
        //     for (uint32_t i = 0; i < itopk_size; ++i) {
        //         uint32_t idx = result_indices[i];
        //         printf("0x%x ", idx);
        //     }
        //     printf("\n");

        //     printf("itopk dists are as follows: ");
        //     for (uint32_t i = 0; i < itopk_size; ++i) {
        //         float dist = result_dists[i];
        //         printf("%f ", dist);
        //     }
        //     printf("\n");
        // }


        // B. Pickup Parents (完全复用)
        if (tid < 32) {
            cagra::device::pickup_next_parents(
                (uint32_t*)terminate_flag, parent_list, result_indices,
                itopk_size, search_width
            );
        }
        __syncthreads();

        // C. Check
        if (*terminate_flag == 1) break;
        // D. Expand (使用 STRIDED 版本)
        // 【核心差异】使用 compute_distance_to_child_nodes_strided
        cagra::device::compute_distance_to_child_nodes_strided(
            result_indices + itopk_size,
            result_dists + itopk_size,
            query_buffer, 
            dataset_ptr, 
            knn_graph, 
            graph_stride,   // 物理宽度 32
            active_degree,  // 逻辑宽度 28 (Local Only)
            dim,
            visited_hash, 
            hash_bitlen, 
            parent_list, 
            search_width,
            result_indices,
            queue_capacity
        );
        __syncthreads();


    }

    // if (tid < 32) {
        if (queue_capacity == 64 && tid < 32)       load_sort_store<2>(result_dists, result_indices, 64);
        else if (queue_capacity == 128 && tid < 32) load_sort_store<4>(result_dists, result_indices, 128);
        else if (queue_capacity == 256 && tid < 32) cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
        else if (queue_capacity == 512 && tid < 32) cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
        else if (queue_capacity == 32 * 32) cagra::radix::load_sort_store(result_dists, result_indices, 1024);
        else if (queue_capacity == 32 * 64) cagra::radix::load_sort_store(result_dists, result_indices, 2048);
        // else if (queue_capacity == 32 * 128) cagra::radix::load_sort_store<128>(result_dists, result_indices, 4096);
        // else if (queue_capacity == 32 * 256) cagra::radix::load_sort_store<256>(result_dists, result_indices, 8192);
        else {
            // 不支持的容量大小
            if (tid == 0) {
                printf(">> [search_kernel_bucket] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
            }
        }
    // }
    __syncthreads();

    uint32_t output_offset = query_id * topk;
    for (uint32_t i = tid; i < topk; i += blockDim.x) {
        uint32_t idx = result_indices[i] & 0x7FFFFFFF;
        float dist = result_dists[i];
        if (result_indices_ptr) result_indices_ptr[output_offset + i] = idx;
        if (result_distances_ptr) result_distances_ptr[output_offset + i] = dist;
    }

    // if (tid == 0) printf("[Bucket Search] Query %u finished in %u iterations.\n", query_id, iter);

    if (tid == 0 && num_executed_iterations) {
        num_executed_iterations[query_id] = iter;
    }
}


__global__ void search_kernel_range(
    uint32_t* result_indices_ptr,       
    float* result_distances_ptr,        
    const float* queries_ptr,           
    const float* dataset_ptr,           
    const uint32_t* knn_graph,          
    const uint32_t* seed_ptr,      
    uint64_t* d_ts,     
    uint32_t num_provided_seeds,        
    uint32_t* num_executed_iterations,  
    
    // --- 核心参数 ---
    uint32_t num_queries,
    size_t num_dataset,
    uint32_t dim,               
    uint32_t graph_stride,      // 图的物理宽度 (32)
    uint32_t active_degree,     // 实际使用的逻辑宽度 (28 - Local Edge)
    uint64_t start_bucket,      // [start_bucket, end_bucket)
    uint64_t end_bucket,
    
    uint32_t topk,              
    uint32_t itopk_size,        
    uint32_t search_width,      
    uint32_t max_iterations,    
    uint32_t num_seeds,         
    uint64_t rand_xor_mask,     
    uint32_t hash_bitlen,
    uint32_t* pre_hashmap,   
    uint32_t queue_capacity     
) {
    auto t_start = clock64();
    // 1. Shared Memory Init (完全复用，代码一样)
    extern __shared__ uint8_t smem[]; 
    size_t offset = 0;

    float* query_buffer = (float*)(smem + offset);
    offset += (dim * sizeof(float) + 15) & ~15;

    uint32_t* visited_hash = nullptr;
    if (hash_bitlen < 14) {
        visited_hash = (uint32_t*)(smem + offset);
        offset += ((1u << hash_bitlen) * sizeof(uint32_t) + 15) & ~15;     
    } else {
        visited_hash = pre_hashmap + (blockIdx.x * (1u << hash_bitlen));
    }

    uint32_t* result_indices = (uint32_t*)(smem + offset);
    offset += (queue_capacity * sizeof(uint32_t) + 15) & ~15;

    float* result_dists = (float*)(smem + offset);
    offset += (queue_capacity * sizeof(float) + 15) & ~15;

    uint32_t* parent_list = (uint32_t*)(smem + offset);
    offset += (search_width * sizeof(uint32_t) + 15) & ~15;

    volatile uint32_t* terminate_flag = (uint32_t*)(smem + offset);

    // 2. Thread Init
    const uint32_t query_id = blockIdx.x;
    if (query_id >= num_queries) return;
    const uint32_t tid = threadIdx.x;

    const float* global_query = queries_ptr + (size_t)query_id * dim;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = global_query[i];
    }
    
    if (tid == 0) *terminate_flag = 0;
    cagra::hashmap::init(visited_hash, hash_bitlen);
    __syncthreads(); 

    // 3. 初始种子阶段
    // 【注意】这里我们假设外部一定提供了足够的桶内种子，或者 num_seeds 设为 0
    // 如果没有种子，我们不能全图随机，必须在桶内随机。
    // 但为了简化，我们复用 init_nodes，并假设上层逻辑保证了种子质量。
    // 如果上层没传种子，这里的兜底随机逻辑是全图的，可能会“跳出桶”。
    // *修正策略*：如果你希望 Kernel 内部支持桶内随机，你需要传 bucket_start 和 bucket_size 进来。
    // 但既然你说了“Host 端直接生成”，那我们就信任 seed_ptr。
    
    const uint32_t* local_seed_ptr = nullptr;
    if (seed_ptr != nullptr) {
        local_seed_ptr = seed_ptr + (size_t)query_id * num_provided_seeds;
    }

    cagra::device::compute_distance_to_init_nodes(
        result_indices, result_dists, query_buffer, dataset_ptr,
        num_dataset, dim, queue_capacity, 
        num_seeds,          
        local_seed_ptr,     
        num_provided_seeds, 
        rand_xor_mask, visited_hash, hash_bitlen
    );
    __syncthreads();

    // 4. 主循环
    uint32_t iter = 0;
    uint32_t hash_reset_iter = 30;

    uint64_t step1_cost = 0;
    uint64_t step2_cost = 0;
    uint64_t step3_cost = 0;

    for (; iter < max_iterations; ++iter) {
        if (iter > 0 && (iter % hash_reset_iter == 0)) {
            cagra::hashmap::init(visited_hash, hash_bitlen);
            __syncthreads();
            cagra::hashmap::restore(visited_hash, hash_bitlen, result_indices, itopk_size);
            __syncthreads();
        }

        // A. Sort (完全复用)
        auto t1 = clock64();
        // if (tid < 32) {
            if (queue_capacity == 64 && tid < 32)       load_sort_store<2>(result_dists, result_indices, 64);
            else if (queue_capacity == 128 && tid < 32) load_sort_store<4>(result_dists, result_indices, 128);
            else if (queue_capacity == 256 && tid < 32) cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
            else if (queue_capacity == 512 && tid < 32) cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
            else if (queue_capacity == 32 * 32) cagra::radix::load_sort_store(result_dists, result_indices, 1024);
            // else if (queue_capacity == 32 * 64) cagra::radix::load_sort_store<64>(result_dists, result_indices, 2048);
            // else if (queue_capacity == 32 * 128) cagra::radix::load_sort_store<128>(result_dists, result_indices, 4096);
            // else if (queue_capacity == 32 * 256) cagra::radix::load_sort_store<256>(result_dists, result_indices, 8192);
            else {
                // 不支持的容量大小
                if (tid == 0) {
                    printf(">> [search_kernel_range] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
                }
            }
            
        // }
        __syncthreads();
        auto t2 = clock64();

        // B. Pickup Parents (完全复用)
        auto t3 = clock64();
        if (tid < 32) {
            cagra::device::pickup_next_parents(
                (uint32_t*)terminate_flag, parent_list, result_indices,
                itopk_size, search_width
            );
        }
        __syncthreads();
        auto t4 = clock64();

        // C. Check
        if (*terminate_flag == 1) break;

        // D. Expand (使用 STRIDED 版本)
        // 【核心差异】使用 compute_distance_to_child_nodes_strided
        auto t5 = clock64();
        cagra::device::compute_distance_to_child_nodes_range(
            result_indices + itopk_size,
            result_dists + itopk_size,
            query_buffer, 
            dataset_ptr, 
            knn_graph, 
            graph_stride,   // 物理宽度 32
            graph_stride,  // 逻辑宽度 28 (Local Only)
            dim,
            visited_hash, 
            hash_bitlen, 
            parent_list, 
            search_width,
            start_bucket,
            end_bucket,
            d_ts
        );
        __syncthreads();
        auto t6 = clock64();

        step1_cost += (t2 - t1);
        step2_cost += (t4 - t3);
        step3_cost += (t6 - t5);
    }

    // 5. 写回 (完全复用)
    if (queue_capacity == 64 && tid < 32)       load_sort_store<2>(result_dists, result_indices, 64);
    else if (queue_capacity == 128 && tid < 32) load_sort_store<4>(result_dists, result_indices, 128);
    else if (queue_capacity == 256 && tid < 32) cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
    else if (queue_capacity == 512 && tid < 32) cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
    else if (queue_capacity == 32 * 32) cagra::radix::load_sort_store(result_dists, result_indices, 1024);
    else {
        // 不支持的容量大小
        if (tid == 0) {
            printf(">> [search_kernel_range] ERROR: Unsupported queue_capacity %u\n", queue_capacity);
        }
    }
    __syncthreads();

    uint32_t output_offset = query_id * topk;
    for (uint32_t i = tid; i < topk; i += blockDim.x) {
        uint32_t idx = result_indices[i] & 0x7FFFFFFF;
        float dist = result_dists[i];
        if (result_indices_ptr) result_indices_ptr[output_offset + i] = idx;
        if (result_distances_ptr) result_distances_ptr[output_offset + i] = dist;
    }

    auto t_end = clock64();

    // if (tid == 0 && query_id <= 5) {
    //     // printf("query %u finished in %u iterations, and queue capacity is %u, and total time cost is %llu.\n", query_id, iter, queue_capacity, step1_cost + step2_cost + step3_cost);
    //     // printf("sort time: %llu, pickup time: %llu, expand time: %llu\n", step1_cost, step2_cost, step3_cost);
    //     // // 转换成百分比再输出一下
    //     // printf("sort perc: %.2f%%, pickup perc: %.2f%%, expand perc: %.2f%%\n", 
    //     //     step1_cost * 100.0 / (step1_cost + step2_cost + step3_cost),
    //     //     step2_cost * 100.0 / (step1_cost + step2_cost + step3_cost),
    //     //     step3_cost * 100.0 / (step1_cost + step2_cost + step3_cost)
    //     // );
    //     printf("total function time cost: %llu\n", t_end - t_start);
    // }
}

} // namespace device
} // namespace cagra