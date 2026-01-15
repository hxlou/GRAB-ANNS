#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace cagra {

// ==========================================
// 1. 核心常量定义
// ==========================================
constexpr int MAX_DEGREE = 512;              // KNN 图最大度数

// ==========================================
// 2. 参数结构体
// ==========================================

struct BuildParams {
    uint32_t intermediate_degree = 128;
    uint32_t graph_degree = 64;
};

struct SearchParams {
    uint32_t itopk_size = 64;
    uint32_t search_width = 5;
    uint32_t min_iterations = 0;
    uint32_t max_iterations = 20;
    uint32_t hash_bitlen = 12;          // 哈希表大小 (2^bitlen)
    static constexpr uint32_t TEAM_SIZE = 32; 
};

// ==========================================
// 3. build & optimize 接口
// ==========================================

/**
 * @brief 使用 FAISS 生成初始 KNN 图 (GPU 计算，CPU 输出)
 * 
 * 流程：FAISS Search (GPU) -> 计算精确距离 (GPU) -> 拷贝回 Host -> 排序/去重 (CPU) -> 写入 h_knn_graph
 * 
 * @param d_dataset  GPU 上的数据集指针 [num_dataset, DIM]
 * @param num_dataset 数据集行数
 * @param dim        数据集维度
 * @param k          每个点找多少个邻居
 * @param h_knn_graph 输出缓冲区 (CPU指针!!!)，类型为 uint32_t，大小 [num_dataset * k]
 */
void generate_knn_graph(const float* d_dataset,
                        size_t num_dataset,
                        uint32_t dim,
                        uint32_t k,
                        uint32_t* h_knn_graph); // <--- 改名并注释为 Host 指针

/**
 * @brief 第一步：优化 - 剪枝 (GPU 计算绕路，CPU 排序)
 * 
 * @param h_knn_graph   [Input] CPU 上的初始 KNN 图 [N, K_in] (例如 K=128)
 * @param h_new_graph   [Output] CPU 上的剪枝后图 [N, K_out] (例如 K=64)
 * @param num_dataset   节点数
 * @param input_degree  输入度数
 * @param output_degree 输出度数
 */
void optimize_prune(const uint32_t* h_knn_graph,
                    uint32_t* h_new_graph,
                    size_t num_dataset,
                    uint32_t input_degree,
                    uint32_t output_degree);

/**
 * @brief [优化第二步] 构建反向图
 * 
 * 遍历输入图，统计每个节点的入边 (Incoming Edges)。
 * 
 * @param h_input_graph   [Input] CPU 上的剪枝后图 [N, degree]
 * @param h_rev_graph     [Output] CPU 上的反向图 [N, degree] (需预分配)
 * @param h_rev_counts    [Output] CPU 上的反向图计数 [N] (需预分配)
 * @param num_dataset     节点数
 * @param degree          度数 (输入输出一致)
 */
void optimize_create_reverse_graph(const uint32_t* h_input_graph,
                                   uint32_t* h_rev_graph,
                                   uint32_t* h_rev_counts,
                                   size_t num_dataset,
                                   uint32_t degree);

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
                           uint32_t degree);

// ==========================================
// 4. search 辅助接口
// ==========================================

void build(const float* d_dataset,
           size_t num_dataset,
           const uint32_t* d_raw_knn_graph, // 输入初始 KNN 图
           BuildParams params,
           uint32_t** d_constructed_graph);

__global__ void cast_u32_to_i64_kernel(const uint32_t* src, int64_t* dst, size_t total_count);

// 2. Search (执行搜索)
void search(const float* d_dataset,
            size_t num_dataset,
            uint32_t dim,
            const uint32_t* d_graph,    // [Input] 构建好的图
            uint32_t graph_degree,      // 图度数
            const float* d_queries,
            int64_t num_queries,
            int64_t k,
            SearchParams params,
            int64_t* d_out_indices, 
            float* d_out_dists,
            const uint32_t* d_seeds = nullptr,
            const uint32_t num_seeds = 0
        );

void insert(const float* d_dataset,     // 旧数据
            size_t num_existing,
            size_t num_new,
            const float* d_new_data,    // 新数据 (分离指针)
            uint32_t* d_graph,          // 图数据 (显存，空间需足够)
            uint32_t* h_graph,          // 图数据 (主机，空间需足够)
            uint32_t graph_degree,
            SearchParams search_params);

} // namespace cagra