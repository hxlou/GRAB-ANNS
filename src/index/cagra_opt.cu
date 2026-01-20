#include "cagra.cuh"
#include "bitonic.cuh"
#include "hashmap.cuh"
#include "config.cuh"
#include "compute_distance.cuh"
#include "search.cuh"
#include "smem_cal.cuh"
#include "insert.cuh"
#include "cagra_opt.cuh"
#include "common.cuh"


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h> // 使用 OpenMP 加速 CPU 排序
#include <random>
#include <chrono>
// FAISS 头文件
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h> 
#include <faiss/impl/AuxIndexStructures.h>

#include "raft_help.cuh"
namespace cagra {

// ===============================
// build
// ===============================

// 简单的伪随机数生成器 (用于随机选择邻居)
__device__ __forceinline__ uint32_t wang_hash(uint32_t seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// =============================================================================
// 精排 Kernel: 计算精确距离并重排序
// =============================================================================
// N_ITEMS_PER_THREAD = K / 32
template <int N_ITEMS_PER_THREAD>
__global__ void refine_and_sort_kernel(
    const float* d_dataset,         // [num_dataset, dim] 原始向量
    const int64_t* d_input_indices, // [num_queries, K] FAISS 输出的粗排索引
    uint32_t* d_output_indices,     // [num_queries, K] 输出精排后的索引 (转为 uint32)
    float* d_output_dists,          // [num_queries, K] 输出精排后的精确距离
    size_t num_dataset,
    uint32_t dim,
    uint32_t K
) {
    // 1. 计算当前 Warp 负责的 Query ID
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t lane_id = threadIdx.x % 32;

    if (warp_id >= num_dataset) return;

    // 当前 Query (也就是当前节点自己) 的向量指针
    const float* query_vec = d_dataset + warp_id * dim;
    
    // 当前 Query 在输入/输出数组中的偏移量
    size_t row_offset = warp_id * K;

    // 2. 寄存器数组：存储分给当前线程的候选点
    float my_dists[N_ITEMS_PER_THREAD];
    uint32_t my_indices[N_ITEMS_PER_THREAD];

    // 3. 加载候选点并计算精确距离
    #pragma unroll
    for (int i = 0; i < N_ITEMS_PER_THREAD; ++i) {
        // 计算当前处理的第 k 个候选
        int k = i * 32 + lane_id;
        
        // 读取 FAISS 返回的索引 (int64)
        int64_t idx_64 = d_input_indices[row_offset + k];
        
        // 默认初始化为最大值 (用于无效点沉底)
        my_dists[i] = 3.40282e38f; // MAX_FLOAT
        my_indices[i] = 0xFFFFFFFF;

        if (k < K && idx_64 >= 0 && idx_64 < num_dataset) {
            uint32_t idx_32 = (uint32_t)idx_64;
            
            // 获取候选向量指针
            const float* candidate_vec = d_dataset + (size_t)idx_32 * dim;
            
            // 【核心】计算精确 L2 距离 (利用 Warp 协作)
            // 注意：这里 calc_l2_dist_1024 是 Warp 级操作，
            // 意味着整个 Warp 现在暂停下来，一起算这 32 个向量的距离吗？
            // 不！calc_l2_dist_1024 是 "Warp 算 1 个对"。
            // 这里我们需要 "Warp 算 32 个对"。
            // 直接调用 calc_l2_dist_1024 在这里是不对的，因为那是 32 线程算 1 个。
            // 这里我们希望每个线程算自己的那 1 个 (或 N 个) 向量的一部分？
            // 不，最高效的方式是：既然 dim=1024，我们还是利用 calc_l2_dist_1024 的逻辑，
            // 但需要循环 32 次（或 N*32 次）来算完所有候选。
            
            // 修正策略：
            // 我们不能并行算 32 个候选的距离，因为 calc_l2_dist_1024 占用了整个 Warp。
            // 我们必须串行循环 K 次，每次算一个，然后把结果分发给持有该候选的线程。
            
            // 但这样太慢了。
            // 更好的策略：既然 dim=1024 刚好是一个 Warp 的倍数。
            // 我们可以让 Lane L 负责维度 [L, L+32, ...] 的计算。
            // 但这需要复杂的转置。
            
            // 回归最稳妥的方式：
            // 复用 cagra::device::calc_l2_dist_1024。
            // 这意味着 Warp 里的 32 个线程，必须**步调一致**地依次计算 K 个候选的距离。
            // 算完第 k 个候选的距离后，只有 `lane_id == k % 32` 的线程保存这个距离。
        }
    }

    // --- 重新设计的距离计算循环 ---
    // 为了正确使用 Warp 级距离算子，必须所有线程一起算同一个候选
    for (int k = 0; k < K; ++k) {
        int64_t idx_64 = d_input_indices[row_offset + k];
        float dist = 3.40282e38f;

        if (idx_64 >= 0 && idx_64 < num_dataset) {
             const float* candidate_vec = d_dataset + (size_t)idx_64 * dim;
             // 全员参与计算
            if (dim == 1024) dist = cagra::device::calc_l2_dist_1024(query_vec, candidate_vec);
            else if (dim == 2048) dist = cagra::device::calc_l2_dist_2048(query_vec, candidate_vec);
            else if (dim == 960) dist = cagra::device::calc_l2_dist_960(query_vec, candidate_vec);
            else if (dim == 256) dist = cagra::device::calc_l2_dist_256(query_vec, candidate_vec);
            else if (dim == 128) dist = cagra::device::calc_l2_dist_128(query_vec, candidate_vec);
            else if (dim == 96) dist = cagra::device::calc_l2_dist_96(query_vec, candidate_vec);
            else {
                // 对于非特殊维度，调用通用版本
                printf("[ERROR] unsupported dimension %u in refine_and_sort_kernel!\n", dim);
            }
        }

        // 分发结果：计算出来的 dist 是广播给全 Warp 的
        // 只有负责该 k 位置的线程才保存它
        int owner_lane = k % 32;
        int reg_idx = k / 32;
        
        if (lane_id == owner_lane) {
            my_dists[reg_idx] = dist;
            my_indices[reg_idx] = (uint32_t)idx_64;
        }
    }

    // 4. 双调排序 (Warp Sort)
    // 升序排列：距离小的在前
    cagra::bitonic::warp_sort<float, uint32_t, N_ITEMS_PER_THREAD>(my_dists, my_indices, true);

    // 5. 写回结果
    #pragma unroll
    for (int i = 0; i < N_ITEMS_PER_THREAD; ++i) {
        int k = i * 32 + lane_id;
        if (k < K) {
            d_output_indices[row_offset + k] = my_indices[i];
            d_output_dists[row_offset + k] = my_dists[i];
        }
    }
}

void refine_search_results(const float* d_dataset,
                           size_t num_dataset,
                           uint32_t dim,
                           const int64_t* d_input_indices, // FAISS 结果
                           uint32_t* d_refined_indices,    // [Output] 精排后的 uint32 ID
                           float* d_refined_dists,         // [Output] 精排后的距离
                           uint32_t k)                     // global_k (e.g. 64)
{
    // std::cout << ">> [CAGRA Global] Refining Search Results (Exact L2 + Sort)..." << std::endl;

    int block_size = 256; // 8 Warps -> 处理 8 个 Query
    int grid_size = (num_dataset + 8 - 1) / 8; // 注意：一个Warp处理一个Query

    // 根据 k 选择模板参数 N
    // k=64 -> N=2, k=128 -> N=4
    if (k <= 64) {
        refine_and_sort_kernel<2><<<grid_size, block_size>>>(
            d_dataset, d_input_indices, d_refined_indices, d_refined_dists,
            num_dataset, dim, k
        );
    } else if (k <= 128) {
        refine_and_sort_kernel<4><<<grid_size, block_size>>>(
            d_dataset, d_input_indices, d_refined_indices, d_refined_dists,
            num_dataset, dim, k
        );
    } else if (k <= 256) {
        // 默认支持到 256
        refine_and_sort_kernel<8><<<grid_size, block_size>>>(
            d_dataset, d_input_indices, d_refined_indices, d_refined_dists,
            num_dataset, dim, k
        );
    } else {
        printf("[ERROR] refine_search_results: Unsupported k=%u (max 256)!\n", k);
    }


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// Remote Edge 更新 Kernel (适配 FAISS int64 输出) step 1
// 逻辑更新：remote edge中，前一半用于连接 临近20% 桶范围内的点，后一半用于全连通的节点
// =============================================================================
__global__ void update_remote_edges_kernel(
    uint32_t* d_graph,              // [N, 32] 最终图
    uint64_t* d_ts,                 // 反查表
    uint32_t* d_global_knn,    // [N, K_global] FAISS 输出是 int64
    const float* d_global_dists,    // [N, K_global] (可选)
    size_t num_dataset,
    uint32_t total_degree,          // 32
    uint32_t local_degree,          // 28
    uint32_t global_k               // e.g. 64
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_dataset) return;

    // 1. 读取本地邻居 (Local Neighbors) 用于查重
    uint32_t* my_graph_row = d_graph + tid * total_degree;
    uint64_t my_timestamps = d_ts[tid];

    uint64_t total_buckets = d_ts[num_dataset - 1] + 1;
    uint64_t fanwei = std::max(1ull, (unsigned long long)total_buckets / 5); // 20% 范围
    uint64_t min_ts = my_timestamps >= fanwei ? my_timestamps - fanwei : 0;
    uint64_t max_ts = my_timestamps + fanwei >= total_buckets ? total_buckets - 1 : my_timestamps + fanwei;

    // 2. 遍历全局候选
    // 直接读取 int64，FAISS 的结果就在显存里
    if (tid == 0) printf("num_dataset: %u and global_k is %u\n", num_dataset, global_k);
    // printf("tid is %d and  my_candidate offset is %d\n", tid, tid * global_k);
    uint32_t* my_candidates = d_global_knn + tid * global_k;
    

    /**
     * nearby_max 最多写入 (total_degree - local_degree) / 2 个
     */
    int nearby_filled = 0;
    int nearby_max = (total_degree - local_degree) / 3;

    for (int k = 0; k < global_k; ++k) {
        uint32_t cand_id = my_candidates[k];

        // 2.1 基础过滤：不能是自己，不能是无效值(-1)
        if (cand_id == (uint32_t)tid || cand_id == 0xFFFFFFFF) continue;

        // 检查：最好和自己的time stamp不相同
        if (d_ts[cand_id] == my_timestamps || (d_ts[cand_id] < min_ts || d_ts[cand_id] > max_ts)) continue;

        // 2.3 写入 Remote 槽位
        my_graph_row[local_degree + nearby_filled] = cand_id;
        nearby_filled++;
        my_candidates[k] = 0xFFFFFFFF; // 标记为已使用

        if (nearby_filled >= nearby_max) break;
    }

    /**
     * nearby_filled 写入之后，剩余的部分全部用于写入 global 范围内的点
     */
    int remote_filled = 0;
    int remote_max = (total_degree - local_degree) - nearby_filled;

    for (int k = 0; k < global_k; ++k) {
        uint32_t cand_id = my_candidates[k];

        // 3.1 基础过滤：不能是自己，不能是无效值(-1)
        if (cand_id == (uint32_t)tid || cand_id == 0xFFFFFFFF) continue;

        // 检查：最好和自己的time stamp不相同
        if (d_ts[cand_id] == my_timestamps || (d_ts[cand_id] >= min_ts && d_ts[cand_id] <= max_ts)) continue;

        // 3.3 写入 Remote 槽位
        my_graph_row[local_degree + nearby_filled + remote_filled] = cand_id;
        remote_filled++;
        my_candidates[k] = 0xFFFFFFFF; // 标记为已使用

        if (remote_filled >= remote_max) break;
    }

    /**
     * 如果 remote_filled 没有填满，则继续从 global_knn 里找
     */
    for (int k = 0; k < global_k && remote_filled < remote_max; ++k) {
        uint32_t cand_id = my_candidates[k];

        // 4.1 基础过滤：不能是自己，不能是无效值(-1)
        if (cand_id == (uint32_t)tid || cand_id == 0xFFFFFFFF) continue;

        // 检查：最好和自己的time stamp不相同
        if (d_ts[cand_id] == my_timestamps) continue;

        // 4.3 写入 Remote 槽位
        my_graph_row[local_degree + nearby_filled + remote_filled] = cand_id;
        remote_filled++;
    }
    

    if (tid < 5) printf("Real remote edges filled: %d and nearby edges filled %d\n", remote_filled, nearby_filled);
}

// =============================================================================
// Step 2: Remote Edge 精修 Kernel (混合策略：一半固定，一半避嫌)
// =============================================================================
__global__ void refine_remote_edges_kernel(
    uint32_t* d_graph,              // [N, total_degree]
    const uint64_t* d_ts,           // [N] 时间戳
    const uint32_t* d_global_knn,    // [N, K_global]
    size_t num_dataset,
    uint32_t total_degree,          
    uint32_t local_degree,          
    uint32_t global_k               
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) printf("Refine Kernel Launched!\n");
    if (tid >= num_dataset) return;

    // 1. 计算数量与划分边界
    int num_remote = total_degree - local_degree; // e.g. 4
    if (num_remote <= 0) return;

    // 前一半固定 (Fixed)，后一半可修改 (Modifiable)
    // e.g. num_remote=4 -> fixed=2, modifiable=2
    // e.g. num_remote=3 -> fixed=1, modifiable=2
    int num_fixed = num_remote / 2;

    // 2. 加载自己的 Remote Edges 到寄存器
    uint32_t my_remotes[64];
    uint32_t* my_graph_row = d_graph + tid * total_degree;
    uint64_t my_ts = d_ts[tid];

    // 我们把所有的 Remote 都读进来，因为查重时需要和 Fixed 的那些边比较
    for (int i = 0; i < num_remote; ++i) {
        my_remotes[i] = my_graph_row[local_degree + i];
    }

    // 3. 构建 Ban List (动态采样邻居)
    // -------------------------------------------------------------------------
    // 限制最大采样数，防止寄存器溢出。
    // 假设 num_samples 最大 20，每人 4 条边，Ban List 128 足够了
    constexpr int MAX_BAN_SIZE = 128;
    uint32_t ban_list[MAX_BAN_SIZE]; 
    int ban_count = 0;

    // 确保采样数不超过实际邻居数，也不超过 Ban List 容量限制
    // (每个邻居贡献 num_remote 条边)
    uint32_t actual_samples = 5;        // 预设采样个数
    if (actual_samples > local_degree) actual_samples = local_degree;
    if (actual_samples * num_remote > MAX_BAN_SIZE) actual_samples = MAX_BAN_SIZE / num_remote;

    // 随机选择起始邻居索引
    uint32_t rand_seed = wang_hash((uint32_t)tid);
    uint32_t start_idx = rand_seed % local_degree;

    // 循环采样 neighbor
    for (uint32_t s = 0; s < actual_samples; ++s) {
        // 简单的环形遍历，保证取到的邻居不重复
        // idx = (start + s) % local_degree
        uint32_t neighbor_idx_in_row = (start_idx + s * 10) % local_degree;
        
        uint32_t nid = my_graph_row[neighbor_idx_in_row];

        if (nid < num_dataset) {
            // 读取该邻居的 Remote Edges
            // 注意：这是非合并内存访问，增加 num_samples 会显著增加延迟
            uint32_t* neighbor_row = d_graph + (size_t)nid * total_degree;
            
            // 手动展开小循环读取
            for (int r = 0; r < num_remote; ++r) {
                uint32_t ban_id = neighbor_row[local_degree + r];
                if (ban_id != 0xFFFFFFFF) {
                    ban_list[ban_count++] = ban_id;
                }
            }
        }
    }

    // 4. 检查冲突并替换 (只针对后一半)
    bool changed = false;
    const uint32_t* my_candidates = d_global_knn + tid * global_k;
    
    // cand_ptr: 从 Step 1 没用到的地方开始找备胎
    // Step 1 贪心填满了 num_remote 个，所以我们从 num_remote 开始找
    int cand_ptr = num_remote; 

    // 【关键修改】循环从 num_fixed 开始，跳过前一半
    for (int r = num_fixed; r < num_remote; ++r) {
        uint32_t current_remote = my_remotes[r];
        if (current_remote == 0xFFFFFFFF) continue;

        // 检查是否在 Ban List 中
        bool conflict = false;
        for (int b = 0; b < ban_count; ++b) {
            if (ban_list[b] == current_remote) {
                conflict = true; 
                break;
            }
        }

        // 如果冲突，尝试从 Candidates 中找一个替换
        if (conflict) {
            while (cand_ptr < global_k) {
                uint32_t replacement = (uint32_t)my_candidates[cand_ptr++];
                
                // --- 验证备胎的合法性 ---
                bool is_valid = true;

                // A. 基础检查
                if (replacement == (uint32_t)tid || replacement >= num_dataset) is_valid = false;

                // B. 是否在现有的 remotes 里 (避免重复)
                // 注意：这里会跟 Fixed (前一半) 和 Modifiable (后一半已填的) 都比较
                // 保证了新选的点不会跟“保留下来的铁杆邻居”重复
                if (is_valid) {
                    for(int k=0; k<num_remote; ++k) {
                        if(my_remotes[k] == replacement) { is_valid = false; break; }
                    }
                }

                // C. 是否在 Ban List 里
                if (is_valid) {
                    for(int k=0; k<ban_count; ++k) {
                        if(ban_list[k] == replacement) { is_valid = false; break; }
                    }
                }

                // D. 时间戳检查 (保持异簇策略)
                if (is_valid) {
                    if (d_ts[replacement] == my_ts) is_valid = false;
                }

                if (is_valid) {
                    // 替换！
                    my_remotes[r] = replacement;
                    changed = true;
                    break; 
                }
            }
        }
    }

    // 5. 写回 Global Memory (只写回修改的那一半即可)
    if (changed) {
        #pragma unroll
        for (int i = num_fixed; i < num_remote; ++i) {
            my_graph_row[local_degree + i] = my_remotes[i];
        }
    }
}

// =============================================================================
// Step 3.1: 构建全局反向图 (只扫描 Remote 区域)
// =============================================================================
__global__ void kern_make_global_rev_graph(
    const uint32_t* d_graph,         // [N, total_degree]
    uint32_t* d_rev_graph,           // [N, 32] (假设反向图最大度数也是32)
    uint32_t* d_rev_counts,          // [N]
    size_t num_dataset,
    uint32_t total_degree,
    uint32_t local_degree,
    uint32_t max_rev_degree          // 反向图记录上限，通常 32
) {
    size_t src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_id >= num_dataset) return;

    // 只遍历 Remote 区域 (从 local_degree 到 total_degree)
    for (uint32_t k = local_degree; k < total_degree; ++k) {
        uint32_t dest_id = d_graph[src_id * total_degree + k];
        
        if (dest_id != 0xFFFFFFFF && dest_id < num_dataset) {
            // 原子添加：记录 src 指向了 dest
            uint32_t pos = atomicAdd(&d_rev_counts[dest_id], 1);
            
            if (pos < max_rev_degree) {
                d_rev_graph[(size_t)dest_id * max_rev_degree + pos] = (uint32_t)src_id;
            }
        }
    }
}

// =============================================================================
// Step 3.2: 注入反向边 (Merge / Inject)
// =============================================================================
__global__ void kern_inject_global_reverse_edges(
    uint32_t* d_graph,               // [N, total_degree] (读/写)
    const uint32_t* d_rev_graph,     // [N, max_rev]
    const uint32_t* d_rev_counts,    // [N]
    size_t num_dataset,
    uint32_t total_degree,
    uint32_t local_degree,
    uint32_t max_rev_degree
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_dataset) return;

    uint32_t num_remote = total_degree - local_degree;
    if (num_remote <= 0) return;

    // 1. 定义区域策略
    // Strong (保留): 前 50% 的 Remote Edge (来自 IVF-PQ 最强关联)
    // Weak (可替换): 后 50% 的 Remote Edge
    int num_strong = num_remote / 2;
    int num_weak = num_remote - num_strong;
    
    // 弱区的起始偏移量
    uint32_t weak_start_offset = local_degree + num_strong;

    // 2. 加载当前节点的所有邻居 (Local + Remote) 到寄存器/局部内存用于查重
    // 我们需要检查反向边是否已经存在于 Local 或 Remote Strong 中
    // 由于 Local 可能很大 (28)，完全放寄存器压力大，我们只加载 Remote Strong 用于查重
    // Local 的查重我们在循环里直接读 Global Memory (虽然慢点，但逻辑简单) 
    // 或者我们假设：反向边如果是 Local 的，那无所谓，我们只关心它别和现有的 Remote 重复。
    
    // 简单起见：读取目前的 Remote Edges 用于查重
    uint32_t current_remotes[32]; // max 32
    uint32_t* my_graph_row = d_graph + tid * total_degree;

    #pragma unroll
    for(int i=0; i<num_remote; ++i) {
        current_remotes[i] = my_graph_row[local_degree + i];
    }

    // 3. 获取反向邻居列表
    uint32_t rev_count = d_rev_counts[tid];
    if (rev_count == 0) return; // 没人指向我，无需注入
    if (rev_count > max_rev_degree) rev_count = max_rev_degree;

    const uint32_t* my_rev_list = d_rev_graph + (size_t)tid * max_rev_degree;

    // 4. 尝试注入
    // 我们遍历反向邻居，试图填入 Weak 区域
    // 优先填空位 (0xFF)，如果没空位，随机替换 Weak 区域的一个
    
    // 伪随机状态
    uint32_t rng_state = wang_hash(tid);

    int weak_fill_ptr = 0; // 当前在 Weak 区填到哪了

    for (int i = 0; i < rev_count; ++i) {
        uint32_t candidate = my_rev_list[i];
        
        // --- 查重 ---
        bool exists = false;
        // A. 查 Remote (寄存器)
        for (int r = 0; r < num_remote; ++r) {
            if (current_remotes[r] == candidate) { exists = true; break; }
        }
        if (exists) continue;

        // B. 查 Local (Global Memory)
        // 这一步虽然慢，但为了保证图的精简，建议加上。
        // 如果 candidate 已经是 Local 邻居，那就没必要浪费 Remote 槽位了
        for (int l = 0; l < local_degree; ++l) {
            if (my_graph_row[l] == candidate) { exists = true; break; }
        }
        if (exists) continue;

        // --- 注入逻辑 ---
        // 目标是 Weak 区域: [num_strong ... num_remote-1]
        // 对应的 graph 索引: local_degree + idx
        
        // 策略：
        // 1. 先找 Weak 区里的空位 (0xFF)
        // 2. 如果没空位，随机替换一个 Weak 区的现有节点
        
        bool injected = false;
        
        // 寻找空位
        for (int w = num_strong; w < num_remote; ++w) {
            if (current_remotes[w] == 0xFFFFFFFF) {
                current_remotes[w] = candidate;
                injected = true;
                break;
            }
        }

        // 如果没空位，随机替换 (Reservoir Sampling 思想的简化版)
        if (!injected && num_weak > 0) {
            // 只有一定概率替换，或者直接替换？
            // 为了增强连通性，反向边通常很有价值，我们强制替换掉 Weak 区的一个
            // 随机选一个 Weak 槽位
            rng_state = wang_hash(rng_state);
            int slot = num_strong + (rng_state % num_weak);
            
            current_remotes[slot] = candidate;
            injected = true;
        }
    }

    // 5. 写回 Global Memory
    // 我们只改动了 Remote 区域，写回即可
    // 注意：只写回 Weak 区域其实就够了，但为了代码简单全写回也没事
    #pragma unroll
    for (int i = 0; i < num_remote; ++i) {
        my_graph_row[local_degree + i] = current_remotes[i];
    }
}

// =============================================================================
// Step 3.2 (New): 交错注入 Remote Edges (支持大度数)
// 策略：Forward[0], Reverse[0], Forward[1], Reverse[1]...
// =============================================================================
__global__ void kern_interleave_remote_edges(
    uint32_t* d_graph,               // [N, total_degree] (读/写)
    const uint32_t* d_rev_graph,     // [N, max_rev]
    const uint32_t* d_rev_counts,    // [N]
    size_t num_dataset,
    uint32_t total_degree,
    uint32_t local_degree,
    uint32_t max_rev_degree
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_dataset) return;

    // 计算实际需要的 Remote 数量
    uint32_t num_remote = total_degree - local_degree;
    if (num_remote <= 0) return;

    // --- 【修改点】定义安全上限 ---
    // 128 足够应对 total_degree=128 且 local=0 的极端情况
    // 如果你的图度数可能更大，请调大这个值
    constexpr int MAX_REMOTE_BUF = 128;

    // 防御性截断：如果配置的度数超过了 Buffer，只处理前 128 个
    if (num_remote > MAX_REMOTE_BUF) num_remote = MAX_REMOTE_BUF;

    // 指向当前节点的行
    uint32_t* my_graph_row = d_graph + tid * total_degree;

    // 1. 读取现有的 Forward Remote Edges (暂存)
    uint32_t fwd_candidates[MAX_REMOTE_BUF]; 
    for(int i=0; i<num_remote; ++i) {
        fwd_candidates[i] = my_graph_row[local_degree + i];
    }

    // 2. 读取 Reverse Remote Edges (暂存)
    uint32_t rev_candidates[MAX_REMOTE_BUF];
    
    uint32_t rev_count = d_rev_counts[tid];
    // 限制读取上限
    if (rev_count > max_rev_degree) rev_count = max_rev_degree;
    // 我们只需要读取足够我们“交错”的数量即可，没必要读几千个
    if (rev_count > MAX_REMOTE_BUF) rev_count = MAX_REMOTE_BUF;

    const uint32_t* my_rev_list = d_rev_graph + (size_t)tid * max_rev_degree;
    for(int i=0; i<rev_count; ++i) {
        rev_candidates[i] = my_rev_list[i];
    }

    // 3. 交错填充 (Interleave Logic)
    uint32_t final_remotes[MAX_REMOTE_BUF];
    int filled_count = 0;

    int f_ptr = 0; // Forward 指针
    int r_ptr = 0; // Reverse 指针
    
    // 循环填空
    while (filled_count < num_remote) {
        bool f_available = (f_ptr < num_remote);
        bool r_available = (r_ptr < rev_count);

        if (!f_available && !r_available) break; // 没数据了

        // --- 尝试填入 Forward ---
        if (f_available && filled_count < num_remote) {
            uint32_t cand = fwd_candidates[f_ptr++];
            
            // 查重 (Check Duplicates)
            bool duplicate = false;
            
            // 1. 和自身/无效值比较
            if (cand == 0xFFFFFFFF || cand == (uint32_t)tid) duplicate = true;

            // 2. 和已填入的 Remote 比较
            // (注意：这里必须遍历，因为是未排序数组)
            if (!duplicate) {
                for(int i=0; i<filled_count; ++i) {
                    if (final_remotes[i] == cand) { duplicate = true; break; }
                }
            }

            // 3. 和 Local Edges 比较 (避免浪费 Remote 槽位)
            // 直接读取 Global Memory，避免占用过多寄存器
            if (!duplicate) {
                for(int i=0; i<local_degree; ++i) {
                    if (my_graph_row[i] == cand) { duplicate = true; break; }
                }
            }

            if (!duplicate) {
                final_remotes[filled_count++] = cand;
            }
        }

        // --- 尝试填入 Reverse ---
        if (r_available && filled_count < num_remote) {
            uint32_t cand = rev_candidates[r_ptr++];
            
            // 查重逻辑同上
            bool duplicate = false;
            if (cand == 0xFFFFFFFF || cand == (uint32_t)tid) duplicate = true;

            if (!duplicate) {
                for(int i=0; i<filled_count; ++i) {
                    if (final_remotes[i] == cand) { duplicate = true; break; }
                }
            }

            if (!duplicate) {
                for(int i=0; i<local_degree; ++i) {
                    if (my_graph_row[i] == cand) { duplicate = true; break; }
                }
            }

            if (!duplicate) {
                final_remotes[filled_count++] = cand;
            }
        }
    }

    // 5. 补齐空位 (Padding)
    while (filled_count < num_remote) {
        final_remotes[filled_count++] = 0xFFFFFFFF;
        printf("Warning: Node %u remote edges not fully filled, padding with 0xFFFFFFFF\n", (uint32_t)tid);
    }

    // 6. 写回 Global Memory
    for(int i=0; i<num_remote; ++i) {
        my_graph_row[local_degree + i] = final_remotes[i];
    }
}

// =============================================================================
// Host 函数: Step 3 (构建反向图 + 注入)
// =============================================================================
void enhance_global_connectivity(size_t num_dataset,
                                 uint32_t* d_graph,
                                 uint32_t total_degree,
                                 uint32_t local_degree)
{
    // std::cout << ">> [CAGRA Global] Step 3: Injecting Reverse Edges..." << std::endl;

    // 1. 准备反向图内存
    uint32_t max_rev_degree = 32; 
    uint32_t* d_rev_graph;
    uint32_t* d_rev_counts;
    
    CUDA_CHECK(cudaMalloc(&d_rev_graph, num_dataset * max_rev_degree * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_counts, num_dataset * sizeof(uint32_t)));
    
    // 初始化
    // rev_graph 不需要全 0xFF，只要计数器对就行，但为了 Debug 方便初始化一下
    // CUDA_CHECK(cudaMemset(d_rev_graph, 0xFF, num_dataset * max_rev_degree * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_rev_counts, 0, num_dataset * sizeof(uint32_t)));

    // 2. 启动构建反向图 Kernel
    int block_size = 256;
    int grid_size = (num_dataset + block_size - 1) / block_size;

    kern_make_global_rev_graph<<<grid_size, block_size>>>(
        d_graph,
        d_rev_graph,
        d_rev_counts,
        num_dataset,
        total_degree,
        local_degree,
        max_rev_degree
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 启动注入 Kernel
    kern_inject_global_reverse_edges<<<grid_size, block_size>>>(
        d_graph,
        d_rev_graph,
        d_rev_counts,
        num_dataset,
        total_degree,
        local_degree,
        max_rev_degree
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 清理
    CUDA_CHECK(cudaFree(d_rev_graph));
    CUDA_CHECK(cudaFree(d_rev_counts));
    
    // std::cout << ">> [CAGRA Global] Connectivity Enhanced." << std::endl;
}


void build_global_remote_edges(const float* d_dataset,
                               size_t num_dataset,
                               uint32_t dim,
                               uint32_t* d_graph,
                               uint64_t* d_ts,
                               uint32_t total_degree,
                               uint32_t local_degree)
{
    // std::cout << ">> [CAGRA Global] Building Remote Edges (Direct GPU FAISS)..." << std::endl;

    // 1. 配置参数
    // 搜索足够多的候选以应对去重 (Top-64)
    uint32_t global_k = (total_degree - local_degree ) * 4;
    global_k = std::max(64u, (global_k + 32) - (global_k % 32)); // 向上取整到32的倍数，至少64
    global_k = std::min(256u, global_k); // 最大256
    
    // IVF-PQ 参数
    int nlist = static_cast<int>(4 * std::sqrt(static_cast<double>(num_dataset)));
    nlist = std::max(1, std::min((int)num_dataset, nlist));
    
    // PQ 配置
    int M = 32; // 子量化器数量 (1024 / 32 = 32维一个子段)
    int nbits = 8;

    // 2. 初始化 FAISS 资源
    faiss::gpu::StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 512); // 512MB 临时显存

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = CUDA_DEVICE_ID; // 确保与当前上下文一致

    // 3. 构建并训练索引
    // 注意：faiss::METRIC_L2
    faiss::gpu::GpuIndexIVFPQ index(&res, dim, nlist, M, nbits, faiss::METRIC_L2, config);
    
    // d_dataset 已经在 GPU 上，直接传
    index.train(num_dataset, d_dataset);
    index.add(num_dataset, d_dataset);

    // 设置探测桶数，平衡速度和精度
    index.nprobe = (std::min(nlist, 100));

    // 4. 执行全量搜索 (Self-Search)
    // 申请输出显存
    int64_t* d_global_indices;
    float* d_global_dists;
    CUDA_CHECK(cudaSetDevice(CUDA_DEVICE_ID));
    CUDA_CHECK(cudaMalloc(&d_global_indices, num_dataset * global_k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_global_dists, num_dataset * global_k * sizeof(float)));

    // d_dataset 既是库也是查询
    index.search(num_dataset, d_dataset, global_k, d_global_dists, d_global_indices);

    // 申请精排后的缓冲区 (uint32 类型，直接适配后续 Kernel)
    uint32_t* d_refined_indices;
    float* d_refined_dists;
    CUDA_CHECK(cudaMalloc(&d_refined_indices, num_dataset * global_k * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_refined_dists, num_dataset * global_k * sizeof(float)));

    // 调用精排函数
    refine_search_results(d_dataset, num_dataset, dim, 
                          d_global_indices, // 输入 (int64)
                          d_refined_indices,    // 输出 (uint32)
                          d_refined_dists,      // 输出 (Exact L2)
                          global_k);

    // 释放原始 FAISS 结果，省点显存
    CUDA_CHECK(cudaFree(d_global_indices));
    CUDA_CHECK(cudaFree(d_global_dists));

    printf("Refinement done! and global k is %u\n", global_k);

    // 5. 启动 Update Kernel
    // 直接使用 d_global_indices (int64*)
    int block_size = 256;
    int grid_size = (num_dataset + block_size - 1) / block_size;

    // 更新remote edge
    std::cout << "Launching update_remote_edges_kernel with grid size " << grid_size << " and block size " << block_size << " and golbal_k " << global_k << std::endl;
    update_remote_edges_kernel<<<grid_size, block_size>>>(
        d_graph,
        d_ts,
        d_refined_indices, // 直接传 uint32 指针
        d_refined_dists,
        num_dataset,
        total_degree,
        local_degree,
        global_k
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // Step 2: 冲突消解 (Refine) - 使用新 Kernel
    // -------------------------------------------------------------
    enhance_global_connectivity(
        num_dataset,
        d_graph,
        total_degree,
        local_degree
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. 清理
    CUDA_CHECK(cudaFree(d_refined_indices));
    CUDA_CHECK(cudaFree(d_refined_dists));
    // index 析构时会自动清理内部显存

    // std::cout << ">> [CAGRA Global] Remote Edges Updated." << std::endl;
}

/**
 * @brief 构建时序分区的 CAGRA 图
 * 
 * 流程：
 * 1. [Local Phase] 遍历每个桶：
 *    a. 构建完整的局部图，度数为 total_degree (32)。
 *    b. 直接拷贝到全局图的对应位置 (覆盖)。
 * 2. [Global Phase] 调用 build_global_remote_edges：
 *    a. 传入 local_degree (28)。
 *    b. Kernel 会从 index 28 开始写入，覆盖掉原有的最后 4 个本地邻居。
 * 
 * @param d_dataset     [N, dim] GPU 数据
 * @param total_num     N
 * @param dim           维度
 * @param d_graph       [N, total_degree] GPU 输出图 (需预分配)
 * @param bucket_sizes  每个时间桶的大小列表
 * @param total_degree  32 (最终图的宽度)
 * @param local_degree  28 (保留的本地边数量)
 */
void build_time_partitioned_graph(const float* d_dataset,
                                  size_t total_num,
                                  uint32_t dim,
                                  uint32_t* d_graph,
                                  uint64_t* d_ts,
                                  uint64_t* h_ts,
                                  const std::vector<size_t>& bucket_sizes,
                                  uint32_t total_degree,   // 32
                                  uint32_t local_degree)   // 28
{
    std::cout << ">> [CAGRA Build] Starting Time-Partitioned Build (Overwrite Strategy) with dim " << dim << " ..." << std::endl;
    std::cout << "local_degree: " << local_degree << ", total_degree: " << total_degree << std::endl;


    // 1. [Local Phase] 逐桶构建 full degree (32) 的图
    size_t current_offset = 0;
    
    // KNN 搜索的 K 值，通常比图度数大，用于剪枝
    uint32_t intermediate_degree = total_degree * 2; 


    // TODO 重构，添加并行度优化
    // 注意到，生成方向图只取决于当前的图，所以我们可以先全量并行的构建出每一个时间戳内部的knn图，后续只需要走一遍流程即可
    // 问题的难点：怎么最大化并行构建所有桶数据的粗略的ann图？
    if (total_num / bucket_sizes.size() > 15000 ) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < bucket_sizes.size(); ++i) {

            std::cout << "Processing bucket " << i << " with size " << bucket_sizes[i] << std::endl;

            size_t b_size = bucket_sizes[i];
            if (b_size == 0) continue;

            // 1.1 准备当前桶的数据指针
            const float* d_bucket_data = d_dataset + current_offset * dim;

            // 1.2 生成 Local KNN (Host)
            std::vector<uint32_t> h_local_knn(b_size * intermediate_degree);
            generate_knn_graph(d_bucket_data, b_size, dim, intermediate_degree, h_local_knn.data());

            // 1.3 执行图优化 (CPU) -> 输出 32 度的图
            std::vector<uint32_t> h_local_graph(b_size * total_degree); // 申请 32 宽度的空间
            std::vector<uint32_t> h_rev_graph(b_size * total_degree);
            std::vector<uint32_t> h_rev_counts(b_size);

            // Step 1: Prune (剪枝到 32)
            optimize_prune(h_local_knn.data(), h_local_graph.data(), 
                        b_size, intermediate_degree, total_degree);

            // Step 2: Reverse (构建 32 度的反向图)
            optimize_create_reverse_graph(h_local_graph.data(), h_rev_graph.data(), 
                                        h_rev_counts.data(), b_size, total_degree);

            // Step 3: Merge (注入反向边)
            optimize_merge_graphs(h_local_graph.data(), h_rev_graph.data(), 
                                h_rev_counts.data(), b_size, total_degree);

            // 此时图中的每个边都需要加上offset
            uint32_t local_offset = static_cast<uint32_t>(current_offset);

            #pragma omp parallel for
            for (size_t idx = 0; idx < b_size * total_degree; ++idx) {
                if (h_local_graph[idx] != 0xFFFFFFFF) {
                    h_local_graph[idx] += local_offset;
                }
            }


            // 1.4 直接拷贝到全局显存 (无需 Memcpy2D，因为宽度匹配)
            // 目标地址
            uint32_t* d_dest_ptr = d_graph + current_offset * total_degree;
            
            // 简单的一维拷贝
            CUDA_CHECK(cudaMemcpy(d_dest_ptr, h_local_graph.data(), 
                                b_size * total_degree * sizeof(uint32_t), 
                                cudaMemcpyHostToDevice));

            // 更新偏移量
            current_offset += b_size;
        }
    } else {
        // 重构，调用raft加速
        std::vector<uint32_t> h_local_knn(total_num * intermediate_degree);
        std::vector<uint32_t> h_global_graph(total_num * total_degree); // 申请 32 宽度的空间
        std::vector<uint32_t> h_global_reverse_graph(total_num * total_degree);
        
        uint32_t* d_global_knn;
        CUDA_CHECK(cudaMalloc(&d_global_knn, total_num * intermediate_degree * sizeof(uint32_t)));

        build_batch_knn_graphs(
            d_dataset,
            dim,
            bucket_sizes,
            intermediate_degree,
            d_global_knn
        );

        // 拷贝回主机
        CUDA_CHECK(cudaMemcpy(h_local_knn.data(), d_global_knn,
                              total_num * intermediate_degree * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_global_knn));
        
        // 主机测更新local index 到 global index
        std::vector<uint32_t> bucket_offsets(bucket_sizes.size() + 1, 0);
        for (size_t i = 1; i <= bucket_sizes.size(); ++i) {
            bucket_offsets[i] = bucket_offsets[i - 1] + bucket_sizes[i - 1];
        }

        #pragma omp parallel for
        for (int i = 0; i < total_num; ++i) {
            uint32_t my_offset = bucket_offsets[h_ts[i]];
            for (int j = 0; j < intermediate_degree; ++j) {
                h_local_knn[i * intermediate_degree + j] += my_offset;
            }
        }

        std::vector<float> h_dataset(total_num * dim);
        CUDA_CHECK(cudaMemcpy(h_dataset.data(), d_dataset,
                              total_num * dim * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // // DEBUG 采样 初始 knn 图，第一行输出他的所有邻居，第二行输出他的对应的距离
        // for (size_t ii = 10; ii < total_num; ii += total_num / 50) {
        //     std::cout << "Sample knn for point " << ii << ": ";
        //     for (size_t j = 0; j < intermediate_degree; ++j) {
        //         std::cout << h_local_knn[ii * intermediate_degree + j] << " ";
        //     }
        //     std::cout << std::endl;
        //     std::cout << "dist with point " << ii << ": ";
        //     for (size_t j = 0; j < intermediate_degree; ++j) {
        //         float dist = 0.0f;
        //         // 直接计算距离即可，获取两个点的坐标
        //         const float* p1 = h_dataset.data() + ii * dim;
        //         const float* p2 = h_dataset.data() + h_local_knn[ii * intermediate_degree + j] * dim;
        //         for (size_t d = 0; d < dim; ++d) {
        //             float diff = p1[d] - p2[d];
        //             dist += diff * diff;
        //         }
        //         std::cout << dist << " ";
        //     }
        //     std::cout << std::endl << std::endl;
        // }

        // prune
        optimize_prune(h_local_knn.data(), h_global_graph.data(), 
                       total_num, intermediate_degree, total_degree);

        // reverse
        std::vector<uint32_t> h_rev_counts(total_num);
        optimize_create_reverse_graph(h_global_graph.data(), h_global_reverse_graph.data(), 
                                      h_rev_counts.data(), total_num, total_degree);

        // merge
        optimize_merge_graphs(h_global_graph.data(), h_global_reverse_graph.data(), 
                              h_rev_counts.data(), total_num, total_degree);

        // 直接拷贝到全局显存 (无需 Memcpy2D，因为宽度匹配)
        CUDA_CHECK(cudaMemcpy(d_graph, h_global_graph.data(),
                              total_num * total_degree * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }


    // 2. [Global Phase] 覆盖 Remote Edges
    // 这一步会找到全局最近邻，并从第 local_degree (28) 列开始写入
    // 覆盖掉刚才构建的最后 4 个本地邻居
    std::cout << ">> [CAGRA Build] Starting Global Remote Edges Construction..." << std::endl;
    build_global_remote_edges(d_dataset, 
                              total_num, 
                              dim, 
                              d_graph,
                              d_ts,
                              total_degree, 
                              local_degree); // 传入 28，表示前 28 个受保护

    std::cout << ">> [CAGRA Build] Construction Complete." << std::endl;
}

void search_opt(const float* d_dataset,
            uint32_t dim,
            size_t num_dataset,
            const uint32_t* d_graph,    
            uint32_t graph_degree,      
            const float* d_queries,
            int64_t num_queries,
            int64_t k,
            SearchParams params,
            int64_t* d_out_indices, 
            float* d_out_dists,
            const uint32_t* d_seeds,
            const uint32_t num_seeds_per_query,
            cudaStream_t stream
){
    if (d_graph == nullptr) {
        throw std::runtime_error("Graph is null!");
    }

    uint32_t topk = static_cast<uint32_t>(k);
    uint32_t itopk_size = std::max(topk, params.itopk_size);
    if (itopk_size < 64) itopk_size = 64; 

    // B. 计算 Shared Memory
    size_t smem_size = cagra::detail::calculate_and_check_smem(
        itopk_size, dim, params.search_width, graph_degree, params.hash_bitlen
    );

    // std::cout << "seme size is (MB)" << smem_size / (1024.0 * 1024.0) << std::endl;

    uint32_t raw_needed = itopk_size + params.search_width * graph_degree;
    uint32_t queue_capacity = std::max(cagra::config::BLOCK_SIZE, 
                                       cagra::detail::next_power_of_2(raw_needed));

    // C. 临时内存 (uint32)
    uint32_t* d_out_indices_u32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_indices_u32, num_queries * topk * sizeof(uint32_t)));
    uint32_t* d_pre_hashmap = nullptr;
    if (params.hash_bitlen > 13) {
        size_t total_hash_size = (1u << params.hash_bitlen) * sizeof(uint32_t) * num_queries;
        // printf("num query is %ld\n", num_queries);
        // printf("total_hash_size is %lu MB\n", total_hash_size / (1024u * 1024u));
        CUDA_CHECK(cudaMalloc(&d_pre_hashmap, total_hash_size));
    }

    // D. 随机种子
    std::random_device rd;
    uint64_t rand_xor_mask = rd(); 
    
    // 目标总种子数：如果没有外部种子，就用 32 个随机；如果有，就取最大值
    uint32_t num_seeds_target = std::max(32u, num_seeds_per_query);
    // 限制：不能超过 queue_capacity 的一半，防止初始填满
    if (num_seeds_target > itopk_size) num_seeds_target = itopk_size;

    // E. 启动 Kernel
    dim3 grid(num_queries);
    dim3 block(cagra::config::BLOCK_SIZE);

    cagra::device::search_kernel<<<grid, block, smem_size, stream>>>(
        d_out_indices_u32,
        d_out_dists,
        d_queries,
        d_dataset,
        d_graph,
        d_seeds,              // 传入外部 seeds
        num_seeds_per_query,  // 外部 seeds 数量
        nullptr, 
        
        // Params
        (uint32_t)num_queries,
        num_dataset,
        dim, 
        graph_degree,
        topk,
        itopk_size,
        params.search_width,
        params.max_iterations,
        num_seeds_target,     // 目标总种子数
        rand_xor_mask,
        params.hash_bitlen,
        d_pre_hashmap,
        queue_capacity
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    // F. 类型转换
    size_t total_elements = num_queries * topk;
    size_t convert_block = 256;
    size_t convert_grid = (total_elements + convert_block - 1) / convert_block;
    
    cast_u32_to_i64_kernel<<<convert_grid, convert_block>>>(
        d_out_indices_u32, 
        d_out_indices, 
        total_elements
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_out_indices_u32));
    if (d_pre_hashmap != nullptr) CUDA_CHECK(cudaFree(d_pre_hashmap));
}

void search_bucket_opt(const float* d_dataset,
                        uint32_t dim,
                       size_t num_dataset,
                       const uint32_t* d_graph,
                       uint32_t total_degree,  // stride (32)
                       uint32_t local_degree,  // active (28)
                       const float* d_queries,
                       int64_t num_queries,
                       int64_t k,
                       SearchParams params,
                       int64_t* d_out_indices, 
                       float* d_out_dists,
                       const uint32_t* d_seeds,
                       const uint32_t num_seeds_per_query,
                       cudaStream_t stream)
{
    if (d_graph == nullptr) {
        throw std::runtime_error("Graph is null!");
    }

    // // debug 输出入参
    // std::cout << "[Bucket Search] num_queries: " << num_queries 
    //           << ", k: " << k 
    //           << ", total_degree: " << total_degree 
    //           << ", local_degree: " << local_degree 
    //           << ", num_seeds_per_query: " << num_seeds_per_query 
    //           << std::endl;


    uint32_t topk = static_cast<uint32_t>(k);
    uint32_t itopk_size = std::max(topk, params.itopk_size);
    if (itopk_size < 128) itopk_size = 128; 

    // B. 计算 Shared Memory
    // 注意：这里使用 local_degree 计算需求，因为我们只扩展这么多邻居
    size_t smem_size = cagra::detail::calculate_and_check_smem(
        itopk_size, dim, params.search_width, total_degree, params.hash_bitlen
    );

    // std::cout << "[Bucket Search] SMEM Size: " << smem_size / 1024.0 << " KB" << std::endl;

    uint32_t raw_needed = itopk_size + params.search_width * total_degree;
    uint32_t queue_capacity = std::max(cagra::config::BLOCK_SIZE, 
                                       cagra::detail::next_power_of_2(raw_needed));

    // C. 临时内存 (uint32 输出 & Global Hashmap)
    uint32_t* d_out_indices_u32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_indices_u32, num_queries * topk * sizeof(uint32_t)));


    uint32_t* d_pre_hashmap = nullptr;
    if (params.hash_bitlen > 13) {
        size_t total_hash_size = (1u << params.hash_bitlen) * sizeof(uint32_t) * num_queries;
        // printf("[Bucket Search] Allocating Global Hashmap: %.2f MB\n", total_hash_size / (1024.0 * 1024.0));
        CUDA_CHECK(cudaMalloc(&d_pre_hashmap, total_hash_size));
        // 【关键】必须初始化，否则 Kernel 内读取全是垃圾数据
        CUDA_CHECK(cudaMemset(d_pre_hashmap, 0xFF, total_hash_size));
    }

    // D. 随机种子
    std::random_device rd;
    uint64_t rand_xor_mask = rd(); 
    
    // 目标总种子数
    uint32_t num_seeds_target = (uint32_t)num_seeds_per_query;
    if (num_seeds_target > itopk_size) num_seeds_target = itopk_size;

    // E. 启动 Kernel (search_kernel_bucket)
    dim3 grid(num_queries);
    dim3 block(cagra::config::BLOCK_SIZE);

    cagra::device::search_kernel_bucket<<<grid, block, smem_size, stream>>>(
        d_out_indices_u32,
        d_out_dists,
        d_queries,
        d_dataset,
        d_graph,
        d_seeds,              
        num_seeds_per_query,  
        nullptr, 
        
        // Params
        (uint32_t)num_queries,
        num_dataset,
        dim, 
        total_degree,   // graph_stride (32)
        local_degree,   // active_degree (28) -> 只搜 Local!
        
        topk,
        itopk_size,
        params.search_width,
        params.max_iterations,
        num_seeds_target,     
        rand_xor_mask,
        params.hash_bitlen,
        d_pre_hashmap,
        queue_capacity
    );
    CUDA_CHECK(cudaGetLastError());
    
    // F. 类型转换 (uint32 -> int64)
    size_t total_elements = num_queries * topk;
    size_t convert_block = 256;
    size_t convert_grid = (total_elements + convert_block - 1) / convert_block;
    
    cast_u32_to_i64_kernel<<<convert_grid, convert_block>>>(
        d_out_indices_u32, 
        d_out_indices, 
        total_elements
    );
    CUDA_CHECK(cudaGetLastError());

    // G. 清理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_out_indices_u32));
    if (d_pre_hashmap != nullptr) CUDA_CHECK(cudaFree(d_pre_hashmap));
}

void search_bucket_range(const float* d_dataset,
                        uint32_t dim,
                       size_t num_dataset,
                       const uint32_t* d_graph,
                       uint64_t* d_ts,
                       uint32_t total_degree,  // stride (32)
                       uint32_t local_degree,  // active (28)
                       const float* d_queries,
                       int64_t num_queries,
                       int64_t k,
                       uint64_t start_bucket,
                       uint64_t end_bucket,
                       SearchParams params,
                       int64_t* d_out_indices, 
                       float* d_out_dists,
                       const uint32_t* d_seeds,
                       const uint32_t num_seeds_per_query)
{
    if (d_graph == nullptr) {
        throw std::runtime_error("Graph is null!");
    }

    uint32_t topk = static_cast<uint32_t>(k);
    uint32_t itopk_size = std::max(topk, params.itopk_size);
    if (itopk_size < 64) itopk_size = 64; 

    // B. 计算 Shared Memory
    // 注意：这里使用 local_degree 计算需求，因为我们只扩展这么多邻居
    size_t smem_size = cagra::detail::calculate_and_check_smem(
        itopk_size, dim, params.search_width, total_degree, params.hash_bitlen
    );

    // std::cout << "[Bucket Search] SMEM Size: " << smem_size / 1024.0 << " KB" << std::endl;

    uint32_t raw_needed = itopk_size + params.search_width * total_degree;
    uint32_t queue_capacity = std::max(cagra::config::BLOCK_SIZE, 
                                       cagra::detail::next_power_of_2(raw_needed));

    // C. 临时内存 (uint32 输出 & Global Hashmap)
    uint32_t* d_out_indices_u32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_indices_u32, num_queries * topk * sizeof(uint32_t)));


    uint32_t* d_pre_hashmap = nullptr;
    if (params.hash_bitlen > 13) {
        size_t total_hash_size = (1u << params.hash_bitlen) * sizeof(uint32_t) * num_queries;
        // printf("[Bucket Search] Allocating Global Hashmap: %.2f MB\n", total_hash_size / (1024.0 * 1024.0));
        CUDA_CHECK(cudaMalloc(&d_pre_hashmap, total_hash_size));
        // 【关键】必须初始化，否则 Kernel 内读取全是垃圾数据
        CUDA_CHECK(cudaMemset(d_pre_hashmap, 0xFF, total_hash_size));
    }

    // D. 随机种子
    std::random_device rd;
    uint64_t rand_xor_mask = rd(); 
    
    // 目标总种子数
    uint32_t num_seeds_target = (uint32_t)num_seeds_per_query;
    if (num_seeds_target > itopk_size) num_seeds_target = itopk_size;

    // E. 启动 Kernel (search_kernel_bucket)
    dim3 grid(num_queries);
    dim3 block(cagra::config::BLOCK_SIZE);

    cagra::device::search_kernel_range<<<grid, block, smem_size>>>(
        d_out_indices_u32,
        d_out_dists,
        d_queries,
        d_dataset,
        d_graph,
        d_seeds,
        d_ts,             
        num_seeds_per_query,  
        nullptr, 
        
        // Params
        (uint32_t)num_queries,
        num_dataset,
        dim, 
        total_degree,   // graph_stride (32)
        local_degree,   // active_degree (28) -> 只搜 Local!
        start_bucket,
        end_bucket,
        
        topk,
        itopk_size,
        params.search_width,
        params.max_iterations,
        num_seeds_target,     
        rand_xor_mask,
        params.hash_bitlen,
        d_pre_hashmap,
        queue_capacity
    );
    CUDA_CHECK(cudaGetLastError());
    
    // F. 类型转换 (uint32 -> int64)
    size_t total_elements = num_queries * topk;
    size_t convert_block = 256;
    size_t convert_grid = (total_elements + convert_block - 1) / convert_block;
    
    cast_u32_to_i64_kernel<<<convert_grid, convert_block>>>(
        d_out_indices_u32, 
        d_out_indices, 
        total_elements
    );
    CUDA_CHECK(cudaGetLastError());

    // G. 清理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_out_indices_u32));
    if (d_pre_hashmap != nullptr) CUDA_CHECK(cudaFree(d_pre_hashmap));
}


/** =====================================================================
 * INSERT
 * =============================================================================
 */

// 交换函数：如果 dist[i] > dist[j]，则交换它们 (升序)
__device__ inline void compare_and_swap(float* s_dists, int64_t* s_indices, int i, int j, bool dir) {
    float dist_i = s_dists[i];
    float dist_j = s_dists[j];
    
    // dir=true (升序), dir=false (降序)
    // Bitonic sort 需要根据阶段交替方向
    // 如果 (dist_i > dist_j) == dir，说明顺序不对，需要交换
    // 这里我们还需要处理 FLT_MAX 的情况，确保它们总是沉到底部
    
    bool swap_condition = (dist_i > dist_j) == dir;
    
    if (swap_condition) {
        // 交换 float
        s_dists[i] = dist_j;
        s_dists[j] = dist_i;
        
        // 交换 int64_t
        int64_t idx_i = s_indices[i];
        int64_t idx_j = s_indices[j];
        s_indices[i] = idx_j;
        s_indices[j] = idx_i;
    }
}

template <int CAPACITY>
__global__ void merge_topk_kernel(
    int64_t* __restrict__ d_indices_local, 
    float*   __restrict__ d_dists_local, 
    int search_k,
    const int64_t* __restrict__ d_indices_tmp, 
    const float*   __restrict__ d_dists_tmp, 
    int tmp_size,
    int n_rows
) {
    // 1. 声明共享内存
    // 大小为 CAPACITY * (sizeof(float) + sizeof(int64_t))
    extern __shared__ char smem[];
    float* s_dists = (float*)smem;
    int64_t* s_indices = (int64_t*)&s_dists[CAPACITY]; // 紧接着 float 存放

    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= n_rows) return;

    // 计算全局偏移
    int local_offset = row * search_k;
    int tmp_offset   = row * tmp_size;
    int total_elements = search_k + tmp_size;

    // -----------------------------------------------------------
    // A. 数据加载 (Load) - 处理边界
    // -----------------------------------------------------------
    // 每个线程负责加载一个或多个元素到 Shared Memory
    // 我们的目标是填满 CAPACITY 长度，不足的部分填 FLT_MAX
    
    for (int i = tid; i < CAPACITY; i += blockDim.x) {
        float val = FLT_MAX;
        int64_t idx = -1;

        if (i < search_k) {
            // 加载 local 部分
            val = d_dists_local[local_offset + i];
            idx = d_indices_local[local_offset + i];
        } 
        else if (i < total_elements) {
            // 加载 tmp 部分
            int offset = i - search_k;
            val = d_dists_tmp[tmp_offset + offset];
            idx = d_indices_tmp[tmp_offset + offset];
        }
        // else: 保持 FLT_MAX (Padding)

        s_dists[i] = val;
        s_indices[i] = idx;
    }

    __syncthreads(); // 必须同步，确保所有数据加载完毕

    // -----------------------------------------------------------
    // B. 双调排序 (Bitonic Sort)
    // -----------------------------------------------------------
    // 这是一个标准的并行 Bitonic Sort 实现
    
    // 外层循环：构建不同长度的双调序列 (2, 4, 8, ... CAPACITY)
    for (unsigned int size = 2; size <= CAPACITY; size <<= 1) {
        
        // 决定当前阶段的排序方向
        // Bitonic Merge 阶段：我们需要构造“波峰波谷”
        // 最终我们需要全升序，所以最后一个阶段 (size=CAPACITY) 所有比较都是升序
        
        // 内层循环：合并双调序列
        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            
            // 确保之前的交换完成
            __syncthreads(); 
            
            // 每个线程处理一对比较
            // 我们需要覆盖 0 ~ CAPACITY/2 的比较对
            for (int i = tid; i < CAPACITY / 2; i += blockDim.x) {
                
                int pos = 2 * i - (i & (stride - 1)); // 映射到比较索引
                int comparator_idx = i; // 仅仅用于判断方向逻辑
                
                // 这种映射方式对于 Bitonic 来说比较 trick，我们换一种更直观的写法：
                // 实际上我们只需要遍历所有下标 j
                // 如果 j & stride == 0，则 j 和 j + stride 比较
                
                // --- 修正后的通用并行 Bitonic 逻辑 ---
                // 每个线程计算它要处理的一对索引 (idx_a, idx_b)
                // 逻辑：所有 index j，如果 (j & stride) == 0，则与 j+stride 比较
                
                // 重新映射：让线程 i 映射到具体的对
                // i 从 0 到 CAPACITY/2 - 1
                // 举例 stride=1: 0->(0,1), 1->(2,3) => idx_a = 2*i
                // 举例 stride=2: 0->(0,2), 1->(1,3), 2->(4,6) ...
                
                int idx_a = (i % stride) + (i / stride) * 2 * stride;
                int idx_b = idx_a + stride;

                // 决定方向：
                // 如果是最后一次合并(size == CAPACITY)，全升序(true)
                // 否则，方向由所在的 size 块决定
                bool dir = ((idx_a & size) == 0); 
                
                compare_and_swap(s_dists, s_indices, idx_a, idx_b, dir);
            }
        }
    }

    __syncthreads(); // 排序完成

    // -----------------------------------------------------------
    // C. 写回 (Store)
    // -----------------------------------------------------------
    // 只写回前 search_k 个结果
    for (int i = tid; i < search_k; i += blockDim.x) {
        d_dists_local[local_offset + i] = s_dists[i];
        d_indices_local[local_offset + i] = s_indices[i];
    }
}

void launch_merge_topk_kernel(
    int64_t* d_indices_local, 
    float*   d_dists_local, 
    int search_k,
    const int64_t* d_indices_tmp, 
    const float*   d_dists_tmp, 
    int tmp_size,
    int n_rows,
    cudaStream_t stream
) {
    // 1. 确定总数据量
    int total_len = search_k + tmp_size;
    
    // 2. 寻找最近的 2 的幂次 (Power of 2)
    // 我们的 Capacity 至少要能容纳 total_len
    int capacity = 1;
    while (capacity < total_len) {
        capacity *= 2;
    }

    // 3. 计算 Shared Memory 大小
    // float array + int64 array
    size_t smem_size = capacity * (sizeof(float) + sizeof(int64_t));
    
    // 4. 配置 Kernel 参数
    // 线程数：对于 Bitonic Sort，一个 Block 最少需要 capacity/2 个线程，
    // 但为了方便加载数据，建议直接设置为 capacity (或 min(capacity, 256))。
    // 这里我们直接用 256 个线程，通常足够处理 capacity=512 的情况 (每个线程处理2个元素)
    int threads_per_block = 256;
    if (capacity < 256) threads_per_block = capacity; // 如果数据很少，线程数减少

    // 5. 模板派发 (Template Dispatch)
    // CUDA Kernel 的 shared memory 大小如果是动态分配，需要在 <<<>>> 第三个参数指定
    // Capacity 必须是编译期常量 (Template)，所以需要 switch-case
    if (capacity <= 64) {
        merge_topk_kernel<64><<<n_rows, threads_per_block, smem_size, stream>>>(
            d_indices_local, d_dists_local, search_k, d_indices_tmp, d_dists_tmp, tmp_size, n_rows
        );
    } else if (capacity <= 128) {
        merge_topk_kernel<128><<<n_rows, threads_per_block, smem_size, stream>>>(
            d_indices_local, d_dists_local, search_k, d_indices_tmp, d_dists_tmp, tmp_size, n_rows
        );
    } else if (capacity <= 256) {
        merge_topk_kernel<256><<<n_rows, threads_per_block, smem_size, stream>>>(
            d_indices_local, d_dists_local, search_k, d_indices_tmp, d_dists_tmp, tmp_size, n_rows
        );
    } else if (capacity <= 512) {
        merge_topk_kernel<512><<<n_rows, threads_per_block, smem_size, stream>>>(
            d_indices_local, d_dists_local, search_k, d_indices_tmp, d_dists_tmp, tmp_size, n_rows
        );
    } else if (capacity <= 1024) {
        merge_topk_kernel<1024><<<n_rows, threads_per_block, smem_size, stream>>>(
            d_indices_local, d_dists_local, search_k, d_indices_tmp, d_dists_tmp, tmp_size, n_rows
        );
    } else {
        // 如果超过 1024，需要报错或者使用更通用的 Global Memory 排序
        // 但根据你的场景 (k=128)，这基本不会发生
        fprintf(stderr, "Error: Capacity %d exceeds supported kernel limits\n", capacity);
    }
}

__global__ void add_offset_kernel(
    int64_t* d_indices, 
    size_t total_elements, 
    int64_t offset
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // 只有有效的索引才加 offset
        // 假设无效索引可能是 -1，如果是 -1 则不处理
        // 但 brute_force search 只要 k <= N 通常不会有 -1
        int64_t val = d_indices[idx];

        // if (val > total_elements) printf("Something wrong here! val is %ld\n", val);

        if (val >= 0) {
            d_indices[idx] = val + offset;
        }
    }
}

void insert(const float* d_dataset,
                     uint32_t* d_graph,
                     const uint64_t* d_ts,
                     const uint32_t* d_seeds,
                     size_t num_existing,
                     size_t num_new,
                     bool use_heuristic,
                     int target_ts,
                     const float* d_queries,
                     uint32_t dim,
                     uint32_t total_degree,
                     uint32_t local_degree,
                     SearchParams params,
                     uint32_t num_seeds_per_query)
{
    if (num_new == 0) return;

    // 1. 准备搜索结果缓冲区
    // 我们需要两份结果：一份给 Local，一份给 Global
    uint32_t search_k = params.itopk_size;
    
    int64_t* d_indices_global;
    float* d_dists_global;
    int64_t* d_indices_local;
    float* d_dists_local;

    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&d_indices_global, num_new * search_k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists_global, num_new * search_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices_local, num_new * search_k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists_local, num_new * search_k * sizeof(float)));


    cudaStream_t stream_global, stream_local;
    CUDA_CHECK(cudaStreamCreate(&stream_global));
    CUDA_CHECK(cudaStreamCreate(&stream_local));

    // printf("starting parallel search...\n");

    #pragma omp parallel sections       // 开启多线程优化
    {
        // -----------------------------------------------------------
        // 2. Global Search (全量搜索，无 Seed)
        // 用于填充 Remote Edges
        // -----------------------------------------------------------
        // 注意：只在 num_existing (老数据) 范围内搜索
        #pragma omp section
        {
            bool use_cagra_opt = true;
            if (use_cagra_opt) {
                search_opt(
                    d_dataset, 
                    dim,
                    num_existing,   // Search Space: Old Data
                    d_graph, 
                    total_degree,   // 使用完整的 32 度进行跳跃
                    d_queries, 
                    (int64_t)num_new, 
                    (int64_t)search_k, 
                    params, 
                    d_indices_global, 
                    d_dists_global, 
                    nullptr,        // 无 Seed -> 随机初始化
                    0,
                    stream_global
                );
            } else {
                // 使用暴力索引，看一下效果如何
                rmm::mr::cuda_memory_resource cuda_mr;
                rmm::mr::set_current_device_resource(&cuda_mr);

                cudaStream_t stream;
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                raft::device_resources global_res(stream);
                rmm::cuda_stream_view stream_view(stream);

                auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(d_dataset, num_existing, dim);
                auto indices_view = raft::make_device_matrix_view<int64_t, int64_t>(d_indices_global, num_new, search_k);
                auto dists_view = raft::make_device_matrix_view<float, int64_t>(d_dists_global, num_new, search_k);
                auto queries_view = raft::make_device_matrix_view<const float, int64_t>(d_queries, num_new, dim);

                cudaDeviceSynchronize();
                auto index = raft::neighbors::brute_force::build(global_res, dataset_view);
                raft::neighbors::brute_force::search(
                    global_res,
                    index,
                    queries_view,
                    indices_view,
                    dists_view
                );

            }
        }

        // auto t11 = std::chrono::high_resolution_clock::now();

        // -----------------------------------------------------------
        // 3. Local Search (带 Seed 搜索)
        // 用于填充 Local Edges
        // -----------------------------------------------------------
        #pragma omp section
        {
            bool use_cagra_bucket = true;
            if (use_cagra_bucket) {
                cagra::search_bucket_opt(
                    d_dataset, 
                    dim,
                    num_existing,   // Search Space: Old Data
                    d_graph, 
                    total_degree,
                    local_degree,
                    d_queries, 
                    (int64_t)num_new, 
                    (int64_t)search_k, 
                    params, 
                    d_indices_local, 
                    d_dists_local, 
                    d_seeds,             // 传入桶内种子
                    num_seeds_per_query,
                    stream_local
                );
            } else {
                rmm::mr::cuda_memory_resource cuda_mr;
                rmm::mr::set_current_device_resource(&cuda_mr);

                // 整理当前桶内的所有数据，我们使用暴力来找到local knn
                size_t offset = target_ts * (size_t)10000;
                float* d_bucket_data;
                size_t bucket_size = 10000 * dim * sizeof(float);
                CUDA_CHECK(cudaMalloc(&d_bucket_data, bucket_size));
                CUDA_CHECK(cudaMemcpyAsync(d_bucket_data, d_dataset + offset * dim,
                                           bucket_size, cudaMemcpyDeviceToDevice, stream_local));
                
                cudaStream_t stream;
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                raft::device_resources local_res(stream);
                rmm::cuda_stream_view stream_view(stream);

                auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(d_bucket_data, 10000, dim);
                auto indices_view = raft::make_device_matrix_view<int64_t, int64_t>(d_indices_local, num_new, search_k);
                auto dists_view = raft::make_device_matrix_view<float, int64_t>(d_dists_local, num_new, search_k);
                auto queries_view = raft::make_device_matrix_view<const float, int64_t>(d_queries, num_new, dim);

                cudaDeviceSynchronize();

                auto index = raft::neighbors::brute_force::build(local_res, dataset_view);
                raft::neighbors::brute_force::search(
                    local_res,
                    index,
                    queries_view,
                    indices_view,
                    dists_view
                );

                // 修正索引偏移
                size_t total_elements = num_new * search_k;
                size_t block_size = 256;
                size_t grid_size = (total_elements + block_size - 1) / block_size;
                int64_t index_offset = static_cast<int64_t>(offset);

                add_offset_kernel<<<grid_size, block_size, 0, stream>>>(
                    d_indices_local,
                    total_elements,
                    index_offset
                );
                CUDA_CHECK(cudaGetLastError());

                // 释放资源
                CUDA_CHECK(cudaFree(d_bucket_data));
                cudaStreamDestroy(stream);
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // printf("finished parallel search.\n");

    // // 输出一下local搜索的结果，第一行index第二行dis，保证对其
    // for (int i = 0; i < std::min((size_t)5, num_new); ++i) {
    //     std::vector<int64_t> h_indices_row(search_k);
    //     std::vector<float> h_dists_row(search_k);
    //     CUDA_CHECK(cudaMemcpyAsync(h_indices_row.data(), d_indices_local + i * search_k,
    //                                search_k * sizeof(int64_t), cudaMemcpyDeviceToHost, stream_local));
    //     CUDA_CHECK(cudaMemcpyAsync(h_dists_row.data(), d_dists_local + i * search_k,
    //                                search_k * sizeof(float), cudaMemcpyDeviceToHost, stream_local));
    //     CUDA_CHECK(cudaStreamSynchronize(stream_local));

    //     std::cout << "target ts is " << target_ts << std::endl;
    //     std::cout << "Local Search Result for Query " << i << ":\nIndices: ";
    //     for (const auto& idx : h_indices_row) {
    //         std::cout << idx << " ";
    //     }
    //     std::cout << "\nDists: ";
    //     for (const auto& dist : h_dists_row) {
    //         std::cout << dist << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // batch 间索引
    // printf("start batch refine...\n");
    if (1) {
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::set_current_device_resource(&cuda_mr);
        int64_t* d_indices_tmp;
        float* d_dists_tmp;
        uint32_t tmp_size = std::min((size_t)num_new, (size_t)search_k / 2);
        CUDA_CHECK(cudaMalloc(&d_indices_tmp, num_new * tmp_size * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_dists_tmp, num_new * tmp_size * sizeof(float)));
        
        cudaStream_t stream_batch;
        cudaStreamCreateWithFlags(&stream_batch, cudaStreamNonBlocking);
        raft::device_resources batch_res(stream_batch);
        rmm::cuda_stream_view stream_view(stream_batch);

        auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(d_queries, num_new, dim);
        auto indices_view = raft::make_device_matrix_view<int64_t, int64_t>(d_indices_tmp, num_new, tmp_size);
        auto dists_view = raft::make_device_matrix_view<float, int64_t>(d_dists_tmp, num_new, tmp_size);

        cudaDeviceSynchronize();

        auto index = raft::neighbors::brute_force::build(batch_res, dataset_view);
        raft::neighbors::brute_force::search(
            batch_res,
            index,
            dataset_view,
            indices_view,
            dists_view
        );

        size_t total_elements = num_new * tmp_size;
        size_t block_size = 256;
        size_t grid_size = (total_elements + block_size - 1) / block_size;
        int64_t offset = static_cast<int64_t>(num_existing);

        add_offset_kernel<<<grid_size, block_size, 0, stream_batch>>>(
            d_indices_tmp,
            total_elements,
            offset
        );
        CUDA_CHECK(cudaGetLastError());

        // 排序
        launch_merge_topk_kernel(
            d_indices_local,
            d_dists_local,
            search_k,
            d_indices_tmp,
            d_dists_tmp,
            tmp_size,
            (int)num_new,
            stream_batch
        );

        // 释放资源
        CUDA_CHECK(cudaFree(d_indices_tmp));
        CUDA_CHECK(cudaFree(d_dists_tmp));
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------
    // 4. Update Topology (更新图结构)
    // -----------------------------------------------------------
    // 调用之前封装好的 update 函数
    // 此时它会同时利用 local knn 和 global knn 填充新节点，
    // 并生成请求更新老节点
    cagra::update_topology_gpu_opt(
        d_graph,
        d_dataset,
        dim,
        target_ts,
        d_ts,
        d_indices_local,  // d_search_indices (Local)
        d_dists_local,    // d_search_dists (Local)
        d_indices_global, // d_search_global (Global)
        d_dists_global,   // d_search_dists (Global)
        num_existing,
        num_new,
        total_degree,
        local_degree,
        search_k,
        search_k,
        use_heuristic
    );

    auto t4 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> search_time = t2 - t1;
    std::chrono::duration<double> batch_time = t3 - t2;
    std::chrono::duration<double> update_time = t4 - t3;

    // std::cout << ">> [CAGRA Algo] Insert Summary: " 
    //           << " Search Time: " << search_time.count() << " s, "
    //           << " Batch Refine Time: " << batch_time.count() << " s, "
    //           << " Update Time: " << update_time.count() << " s."
    //           << std::endl;

    // 5. 清理
    CUDA_CHECK(cudaFree(d_indices_global));
    CUDA_CHECK(cudaFree(d_dists_global));
    CUDA_CHECK(cudaFree(d_indices_local));
    CUDA_CHECK(cudaFree(d_dists_local));

    // std::cout << ">> [CAGRA Algo] Insert Complete." << std::endl;
}

} // namespace cagra