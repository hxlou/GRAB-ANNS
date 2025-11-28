#include "cagra.cuh"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h> // 使用 OpenMP 加速 CPU 排序

// FAISS 头文件
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h> 
#include <faiss/impl/AuxIndexStructures.h>

namespace cagra {

// =============================================================================
// Kernel: 计算精确 L2 距离 (Refine 核心)
// =============================================================================
__global__ void compute_exact_distances_kernel(const float* d_dataset,     // [N, dim]
                                               const int64_t* d_indices,   // [batch, search_k] FAISS 搜出来的候选ID
                                               float* d_exact_dists,       // [batch, search_k] 输出精确距离
                                               size_t num_dataset,
                                               int dim,
                                               int batch_size,
                                               int search_k,
                                               size_t offset)              // 当前 batch 在全集中的偏移量
{
    // 扁平化线程索引
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size * search_k) return;

    int local_q_idx = tid / search_k;      // 当前 batch 内的第几个查询
    // int cand_pos = tid % search_k;      // 候选列表中的第几个 (未使用)
    
    size_t global_q_idx = offset + local_q_idx;   // 全局查询 ID
    int64_t neighbor_id = d_indices[tid];         // 候选邻居的全局 ID

    // 如果 FAISS 返回 -1 (无效)，距离设为无穷大
    if (neighbor_id < 0 || neighbor_id >= num_dataset) {
        d_exact_dists[tid] = 1e30f; // Max float
        return;
    }

    // 计算 L2 距离
    float dist_sq = 0.0f;
    const float* vec_q = d_dataset + global_q_idx * dim;
    const float* vec_n = d_dataset + neighbor_id * dim;

    for (int d = 0; d < dim; ++d) {
        float diff = vec_q[d] - vec_n[d];
        dist_sq += diff * diff;
    }

    d_exact_dists[tid] = dist_sq;
}

// =============================================================================
// 主函数: 使用 FAISS IVF-PQ 构建 KNN 图 (输出到 CPU)
// =============================================================================
void generate_knn_graph(const float* d_dataset,
                        size_t num_dataset,
                        uint32_t dim,
                        uint32_t k,
                        uint32_t* h_knn_graph) // 输出到 CPU
{
    // 1. 初始化 FAISS GPU 资源
    faiss::gpu::StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 512); 

    // 2. 配置参数
    int nlist = static_cast<int>(4 * std::sqrt(static_cast<double>(num_dataset)));
    nlist = std::max(1, std::min((int)num_dataset, nlist));
    int M = 32; 
    int nbits = 8;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = 0; 
    
    faiss::gpu::GpuIndexIVFPQ index(&res, dim, nlist, M, nbits, faiss::METRIC_L2, config);

    // 3. 训练与添加
    index.train(num_dataset, d_dataset);
    index.add(num_dataset, d_dataset);
    
    // 设置探测数
    // 新版 FAISS 推荐直接设置 public 成员
    index.nprobe = std::min(nlist, 20); 

    // =========================================================
    // 4. Batch 搜索与精排
    // =========================================================
    
    // 扩大搜索范围以提高 Recall
    int search_k = k * 2; 
    search_k = std::min((size_t)search_k, num_dataset);

    int max_batch_size = 1024; // 每个 Batch 处理 1024 个 Query

    // GPU 临时内存
    int64_t* d_search_indices;
    float* d_approx_dists; // FAISS 需要这个缓冲区，虽然我们后面会重新算
    float* d_exact_dists;  // 我们自己算的精确距离

    cudaMalloc(&d_search_indices, max_batch_size * search_k * sizeof(int64_t));
    cudaMalloc(&d_approx_dists,   max_batch_size * search_k * sizeof(float));
    cudaMalloc(&d_exact_dists,    max_batch_size * search_k * sizeof(float));

    // CPU 临时内存 (用于接收 GPU 数据进行处理)
    // 使用 vector 自动管理内存
    std::vector<int64_t> h_search_indices(max_batch_size * search_k);
    std::vector<float>   h_exact_dists(max_batch_size * search_k);

    // 遍历 Batch
    for (size_t offset = 0; offset < num_dataset; offset += max_batch_size) {
        int current_batch = std::min((size_t)max_batch_size, num_dataset - offset);

        // A. FAISS 搜索 (GPU)
        index.search(current_batch, d_dataset + offset * dim, search_k, d_approx_dists, d_search_indices);

        // B. 计算精确距离 (GPU Kernel)
        int total_threads = current_batch * search_k;
        int block_size = 256;
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        compute_exact_distances_kernel<<<grid_size, block_size>>>(
            d_dataset,
            d_search_indices,
            d_exact_dists,
            num_dataset,
            dim,
            current_batch,
            search_k,
            offset // 当前 batch 偏移
        );
        cudaDeviceSynchronize();

        // C. 数据拷回 CPU
        cudaMemcpy(h_search_indices.data(), d_search_indices, current_batch * search_k * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_exact_dists.data(), d_exact_dists, current_batch * search_k * sizeof(float), cudaMemcpyDeviceToHost);

        // D. CPU 精排与过滤
        // OpenMP 并行处理当前 Batch 内的所有 Query
        #pragma omp parallel for 
        for (int i = 0; i < current_batch; ++i) {
            size_t global_query_id = offset + i;
            
            // 收集当前 Query 的所有候选 (distance, index)
            std::vector<std::pair<float, int64_t>> candidates;
            candidates.reserve(search_k);

            for (int j = 0; j < search_k; ++j) {
                int flat_idx = i * search_k + j;
                int64_t idx = h_search_indices[flat_idx];
                float dist = h_exact_dists[flat_idx];

                // 过滤无效索引
                if (idx >= 0) {
                    candidates.push_back({dist, idx});
                }
            }

            // 排序 (按精确距离从小到大)
            std::sort(candidates.begin(), candidates.end());

            // 填充结果到最终的 CPU 输出缓冲区
            uint32_t* output_ptr = h_knn_graph + global_query_id * k;
            int filled = 0;

            for (const auto& pair : candidates) {
                if (filled >= k) break;
                
                // 去除自身 (self-loop)
                if (pair.second != (int64_t)global_query_id) {
                    output_ptr[filled] = static_cast<uint32_t>(pair.second);
                    filled++;
                }
            }

            // 如果没填满，补 0xFFFFFFFF
            while (filled < k) {
                output_ptr[filled] = 0xFFFFFFFF;
                filled++;
            }
        }

        // 打印进度
        if ((offset / max_batch_size) % 10 == 0) {
            // std::cout << "Processed " << offset << " / " << num_dataset << "\r" << std::flush;
        }
    }
    // std::cout << std::endl;

    // 5. 清理 GPU 临时资源
    cudaFree(d_search_indices);
    cudaFree(d_approx_dists);
    cudaFree(d_exact_dists);
}


/**
 * @brief 剪枝 Kernel
 * 
 * 计算每条边的绕路计数 (Detour Count)。
 * 线程配置：1个 Block (32 threads) 处理 1 个节点。
 */
__global__ void kern_prune(const uint32_t* knn_graph,      // [N, K]
                           uint8_t* detour_count,          // [N, K]
                           size_t num_dataset,
                           uint32_t graph_degree,
                           size_t offset)                  // Batch 偏移
{
    // 共享内存：存储当前节点的边绕路计数
    __shared__ uint32_t smem_num_detour[MAX_DEGREE];

    // 当前 Block 处理的全局节点 ID
    const size_t nid = blockIdx.x + offset;
    
    if (nid >= num_dataset) return;
    const size_t iA = nid;

    // 1. 初始化共享内存
    // threadIdx.x 范围 0-31，步长 32，覆盖所有 graph_degree (最大 512)
    for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
        smem_num_detour[k] = 0;
    }
    __syncthreads();

    // 2. 寻找三角形 A->D->B
    // 外层循环：遍历 A 的邻居 D (索引 kAD)
    // 条件：kAD < graph_degree - 1，因为 D 必须在 B 之前 (距离更近)
    for (uint32_t kAD = 0; kAD < graph_degree - 1; kAD++) {
        // 获取 D 的 ID
        const uint32_t iD = knn_graph[kAD + ((size_t)graph_degree * iA)];
        
        // 跳过无效 ID
        if (iD == 0xFFFFFFFF) continue;

        // 内层循环 (Warp 并行)：遍历 D 的邻居，看有没有 B
        for (uint32_t kDB = threadIdx.x; kDB < graph_degree; kDB += blockDim.x) {
            
            // iB_candidate 是 D 的邻居
            const uint32_t iB_candidate = knn_graph[kDB + ((size_t)graph_degree * iD)];

            // 检查 A 的邻居列表中是否有这个点 (且要在 D 后面)
            for (uint32_t kAB = kAD + 1; kAB < graph_degree; kAB++) {
                const uint32_t iB = knn_graph[kAB + ((size_t)graph_degree * iA)];
                
                // 发现三角形 A->D->B 且 dist(A,D) < dist(A,B)
                if (iB == iB_candidate) {
                    atomicAdd(&smem_num_detour[kAB], 1);
                    break; // 只要有一条路径就算绕路
                }
            }
        }
        __syncthreads();
    }

    // 3. 写回全局内存
    for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
        uint32_t count = smem_num_detour[k];
        detour_count[k + ((size_t)graph_degree * iA)] = (count > 255) ? 255 : (uint8_t)count;
    }
}

/**
 * @brief 第一步优化：剪枝 (Pruning)
 * 
 * 逻辑：
 * 1. (GPU) 计算每条边的绕路数 (Detour Count)。
 * 2. (CPU) 根据绕路数从小到大排序，保留前 output_degree 个邻居。
 *    如果绕路数相同，保留原始顺序（距离更近优先）。
 */
void optimize_prune(const uint32_t* h_knn_graph,
                    uint32_t* h_new_graph,
                    size_t num_dataset,
                    uint32_t input_degree,
                    uint32_t output_degree)
{
    std::cout << ">> [Optimize Step 1] Pruning Graph (Batched)..." << std::endl;
    
    // 检查 Shared Memory 限制 (Kernel 中定义的 MAX_DEGREE 为 512)
    if (input_degree > MAX_DEGREE) {
        throw std::runtime_error("Input degree too large for prune kernel (max 512).");
    }

    // ==========================================
    // 1. 准备 GPU 资源
    // ==========================================
    uint32_t* d_input_graph;
    uint8_t* d_detour_count;

    // 申请全量显存 (knn_graph 在 Kernel 中需要随机访问，无法分片)
    CUDA_CHECK(cudaMalloc(&d_input_graph, num_dataset * input_degree * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_detour_count, num_dataset * input_degree * sizeof(uint8_t)));

    // ==========================================
    // 2. 拷贝输入图到 GPU & 初始化输出
    // ==========================================
    std::cout << "   Copying input graph to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_input_graph, h_knn_graph, 
                          num_dataset * input_degree * sizeof(uint32_t), 
                          cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_detour_count, 0, num_dataset * input_degree * sizeof(uint8_t)));

    // ==========================================
    // 3. 分 Batch 执行 Kernel
    // ==========================================
    // 设定 Batch Size: 256K 个节点。防止 GPU TDR (超时重置)
    const size_t batch_size = 256 * 1024;
    
    // 线程配置: 1个 Warp (32 threads) 处理 1 个节点
    dim3 threads(32, 1, 1); 

    for (size_t offset = 0; offset < num_dataset; offset += batch_size) {
        // 计算当前 Batch 的实际大小
        size_t current_batch = std::min(batch_size, num_dataset - offset);
        
        // Grid Size = 当前 Batch 的节点数
        dim3 blocks(current_batch, 1, 1);

        // 启动 Kernel (传入 offset)
        kern_prune<<<blocks, threads>>>(d_input_graph, 
                                        d_detour_count, 
                                        num_dataset, 
                                        input_degree,
                                        offset);
        
        // 简单的进度提示
        if ((offset / batch_size) % 10 == 0) {
             // std::cout << "   Processing batch offset: " << offset << " / " << num_dataset << "\r" << std::flush;
        }
    }
    // 等待所有 Kernel 执行完毕
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "   Kernel execution finished." << std::endl;

    // ==========================================
    // 4. 将绕路计数 (Detour Count) 拷回 CPU
    // ==========================================
    std::cout << "   Copying detour counts back to CPU..." << std::endl;
    std::vector<uint8_t> h_detour_count(num_dataset * input_degree);
    
    CUDA_CHECK(cudaMemcpy(h_detour_count.data(), d_detour_count, 
                          num_dataset * input_degree * sizeof(uint8_t), 
                          cudaMemcpyDeviceToHost));

    // ==========================================
    // 5. CPU 端的筛选逻辑 (Rank-based Pruning)
    // ==========================================
    std::cout << "   Ranking and pruning on CPU..." << std::endl;
    
    // 使用 OpenMP 并行加速循环
    #pragma omp parallel for
    for (size_t i = 0; i < num_dataset; ++i) {
        // 指针定位
        const uint32_t* input_ptr = h_knn_graph + i * input_degree;
        const uint8_t* count_ptr = h_detour_count.data() + i * input_degree;
        uint32_t* output_ptr = h_new_graph + i * output_degree;

        // 创建索引数组 [0, 1, ..., input_degree-1]
        std::vector<int> indices(input_degree);
        for(int k=0; k<input_degree; ++k) indices[k] = k;

        // 核心排序逻辑：
        // 1. Detour Count 小的优先 (越小越重要，说明没有绕路替代方案)
        // 2. Count 相等时，保留原始顺序 (原始图按距离排序，所以保留距离近的)
        std::stable_sort(indices.begin(), indices.end(), 
            [&](int a, int b) {
                return count_ptr[a] < count_ptr[b];
            }
        );

        // 填充输出图 (只取前 output_degree 个)
        for (uint32_t k = 0; k < output_degree; ++k) {
            int original_idx = indices[k];
            output_ptr[k] = input_ptr[original_idx];
        }
    }

    // ==========================================
    // 6. 清理 GPU 资源
    // ==========================================
    CUDA_CHECK(cudaFree(d_input_graph));
    CUDA_CHECK(cudaFree(d_detour_count));
    
    std::cout << ">> Pruning Finished." << std::endl;
}

/**
 * @brief 构建反向图 Kernel
 * 
 * 逻辑：
 * 线程 i 负责处理源节点 src_id = i。
 * 假如 input_graph[src_id] 的某一个邻居是 dest_id。
 * 意味着 src_id 指向 dest_id。
 * 那么在反向图中，dest_id 的列表里应该包含 src_id。
 * 
 * @param dest_nodes      [N] 输入数据（当前处理的那一列邻居）
 * @param rev_graph       [N, K] 反向图 (输出)
 * @param rev_graph_count [N] 反向图的当前计数 (输出)
 * @param graph_size      节点数 N
 * @param degree          反向图的最大度数 K
 */
__global__ void kern_make_rev_graph(const uint32_t* dest_nodes,     
                                    uint32_t* rev_graph,            
                                    uint32_t* rev_graph_count,  
                                    uint32_t graph_size,
                                    uint32_t degree)
{
    // Grid-Stride Loop: 允许任意 Grid 大小处理任意数据量
    const uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
    const uint32_t tnum = blockDim.x * gridDim.x;

    for (uint32_t src_id = tid; src_id < graph_size; src_id += tnum) {
        // src_id 指向了 dest_id
        const uint32_t dest_id = dest_nodes[src_id];

        // 过滤无效节点 (比如 padding 的 0xFFFFFFFF 或者超过范围的 ID)
        if (dest_id >= graph_size) continue;

        // 核心：原子加，抢占写入位置
        // old_count = rev_graph_count[dest_id]++
        const uint32_t pos = atomicAdd(&rev_graph_count[dest_id], 1);
        
        // 如果反向列表没满，就写入 src_id
        if (pos < degree) { 
            rev_graph[pos + ((size_t)degree * dest_id)] = src_id; 
        }
    }
}

/**
 * @brief 第二步优化：构建反向图
 * 
 * @param h_input_graph   [Input] CPU 上的输入图 (剪枝后的图) [N, degree]
 * @param h_rev_graph     [Output] CPU 上的反向图 [N, degree]
 * @param h_rev_counts    [Output] CPU 上的反向图计数 [N]
 * @param num_dataset     节点数 N
 * @param degree          图的度数 K (输入和输出度数一致)
 */
void optimize_create_reverse_graph(const uint32_t* h_input_graph,
                                   uint32_t* h_rev_graph,
                                   uint32_t* h_rev_counts,
                                   size_t num_dataset,
                                   uint32_t degree)
{
    std::cout << ">> [Optimize Step 2] Creating Reverse Graph..." << std::endl;

    // ==========================================
    // 1. 准备 GPU 资源
    // ==========================================
    uint32_t* d_rev_graph;
    uint32_t* d_rev_counts;
    uint32_t* d_dest_nodes; // 临时缓冲区，存一列数据

    CUDA_CHECK(cudaMalloc(&d_rev_graph, num_dataset * degree * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_counts, num_dataset * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dest_nodes, num_dataset * sizeof(uint32_t)));

    // ==========================================
    // 2. 初始化
    // ==========================================
    // 反向图初始化为 0xFFFFFFFF (无效值)
    CUDA_CHECK(cudaMemset(d_rev_graph, 0xFF, num_dataset * degree * sizeof(uint32_t)));
    // 计数器初始化为 0
    CUDA_CHECK(cudaMemset(d_rev_counts, 0, num_dataset * sizeof(uint32_t)));

    // CPU 端临时 buffer，用于提取一列
    std::vector<uint32_t> h_col_buffer(num_dataset);

    // ==========================================
    // 3. 纵向切片遍历 (按列处理)
    // ==========================================
    // 每次处理所有节点的第 k 个邻居
    
    // Kernel 配置
    int block_size = 256;
    int grid_size = (num_dataset + block_size - 1) / block_size;
    // 限制一下 grid 大小，防止超大
    grid_size = std::min(grid_size, 2048); 

    for (uint32_t k = 0; k < degree; ++k) {
        
        // [CPU] 提取第 k 列
        // 即：所有节点 i 的第 k 个邻居
        #pragma omp parallel for
        for (size_t i = 0; i < num_dataset; ++i) {
            h_col_buffer[i] = h_input_graph[i * degree + k];
        }

        // [Copy] 拷贝这一列到 GPU
        CUDA_CHECK(cudaMemcpy(d_dest_nodes, h_col_buffer.data(), 
                              num_dataset * sizeof(uint32_t), 
                              cudaMemcpyHostToDevice));

        // [Kernel] 执行反向插入
        kern_make_rev_graph<<<grid_size, block_size>>>(
            d_dest_nodes,
            d_rev_graph,
            d_rev_counts,
            num_dataset,
            degree
        );

        // 简单的进度打印
        if (k % 10 == 0) {
            // std::cout << "   Processing column: " << k << " / " << degree << "\r" << std::flush;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << std::endl;

    // ==========================================
    // 4. 结果拷回 CPU
    // ==========================================
    std::cout << "   Copying results back to CPU..." << std::endl;
    
    CUDA_CHECK(cudaMemcpy(h_rev_graph, d_rev_graph, 
                          num_dataset * degree * sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost));
                          
    CUDA_CHECK(cudaMemcpy(h_rev_counts, d_rev_counts, 
                          num_dataset * sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost));

    // ==========================================
    // 5. 清理
    // ==========================================
    CUDA_CHECK(cudaFree(d_rev_graph));
    CUDA_CHECK(cudaFree(d_rev_counts));
    CUDA_CHECK(cudaFree(d_dest_nodes));

    std::cout << ">> Reverse Graph Created." << std::endl;
}

// 辅助函数：在数组中查找元素的位置
// 如果找到，返回 index；如果没找到，返回 size
inline uint32_t find_pos_in_array(const uint32_t* arr, uint32_t size, uint32_t target) {
    for (uint32_t i = 0; i < size; ++i) {
        if (arr[i] == target) return i;
    }
    return size;
}

/**
 * @brief 第三步优化：合并图 (注入反向边)
 * 
 * 逻辑：
 * 1. 保护前 degree/2 个邻居。
 * 2. 遍历反向图，将反向邻居插入到 degree/2 的位置。
 * 3. 数组右移，丢弃末尾元素。
 */
void optimize_merge_graphs(uint32_t* h_graph,           // [In/Out] 主图 (剪枝后的图)
                           const uint32_t* h_rev_graph, // [In] 反向图
                           const uint32_t* h_rev_counts,// [In] 反向计数
                           size_t num_dataset,
                           uint32_t degree)
{
    // 保护区大小：前一半
    const uint32_t num_protected = degree / 2;

    // OpenMP 并行处理每个节点
    #pragma omp parallel for
    for (size_t j = 0; j < num_dataset; ++j) {
        // 当前节点 j 的邻居列表指针
        uint32_t* graph_ptr = h_graph + j * degree;
        
        // 获取有多少节点指向 j
        uint32_t rev_count = h_rev_counts[j];
        // 限制处理数量，不能超过度数 (虽然一般 rev_graph 也没存那么多)
        if (rev_count > degree) rev_count = degree;

        const uint32_t* rev_ptr = h_rev_graph + j * degree;

        // 遍历每一个指向 j 的节点 i
        // 倒序遍历还是正序遍历在随机图上影响不大，这里采用正序
        for (uint32_t k = 0; k < rev_count; ++k) {
            uint32_t i = rev_ptr[k];

            // 过滤无效节点
            if (i >= num_dataset) continue;
            // 理论上 i 不应该等于 j (无自环)，稍微防一下
            if (i == (uint32_t)j) continue;

            // 1. 检查 i 是否已经在 j 的邻居列表中
            uint32_t pos = find_pos_in_array(graph_ptr, degree, i);

            // 2. 如果 i 已经在保护区 (前一半)，说明它非常重要，不需要动
            if (pos < num_protected) {
                continue;
            }

            // 3. 计算需要移动的元素数量
            // 我们要把 i 插入到 num_protected 的位置
            // 需要把 [num_protected, ...] 的元素往后挪
            uint32_t shift_len = 0;

            if (pos < degree) {
                // 情况 A: i 已经在列表中，但在非保护区 (pos >= num_protected)
                // 我们把它“提拔”到非保护区的首位
                // 需要挪动 [num_protected, pos - 1] 这一段
                shift_len = pos - num_protected;
            } else {
                // 情况 B: i 不在列表中
                // 我们把它强行插入到非保护区首位
                // 需要挪动 [num_protected, degree - 2] 这一段 (最后一个元素被丢弃)
                shift_len = degree - num_protected - 1;
            }

            // 4. 执行数组移动 (memmove 处理重叠区域是安全的)
            // 目标: graph_ptr + num_protected + 1
            // 源:   graph_ptr + num_protected
            if (shift_len > 0) {
                std::memmove(graph_ptr + num_protected + 1,
                             graph_ptr + num_protected,
                             shift_len * sizeof(uint32_t));
            }

            // 5. 插入新邻居
            graph_ptr[num_protected] = i;
        }
    }

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


} // namespace cagra