#pragma once

#include "index/cagra.cuh"
#include "vmm_allocator.cuh"
#include <string>
#include <vector>
#include <memory>

namespace cagra {

class CagraIndex {
public:
    /**
     * @brief 初始化索引管理器
     * @param dim 向量维度
     * @param graph_degree 最终 CAGRA 图的度数 (默认 32)
     * @param vmm_max_bytes VMM 允许的最大显存占用
     */
    CagraIndex(uint32_t dim, 
               uint32_t graph_degree = 32, 
               size_t vmm_max_bytes = 20ULL * 1024 * 1024 * 1024);

    ~CagraIndex();

    /**
     * @brief [阶段1] 积累数据
     * 仅将数据追加到内部 Host 缓存中，不触发 GPU 操作。
     * 通常用于初始化阶段的数据加载。
     */
    void add(size_t num_vectors, const float* add_vectors);

    /**
     * @brief [阶段2] 全量构建
     * 使用 add() 积累的所有数据，在 GPU 上从零构建索引。
     * 会自动处理 VMM 分配和数据同步。
     */
    void build();

    /**
     * @brief [阶段3] 增量插入
     * 直接插入新数据并实时更新索引。
     * 1. 追加到 Host/Device 数据区
     * 2. 执行搜索寻找邻居
     * 3. 更新图结构
     * 
     * @param new_vectors 新增数量
     * @param insert_vectors 新数据指针
     */
    void insert(size_t new_vectors, const float* insert_vectors);

    /**
     * @brief 向量查询
     */
    void query(const float* host_queries, 
               size_t num_queries, 
               int k, 
               int64_t* host_indices, 
               float* host_dists);

    void setBuildParams(uint32_t inter_degree, uint32_t graph_degree) {
        build_params_.intermediate_degree = inter_degree;
        build_params_.graph_degree = graph_degree;
    };

    void setQueryParams(uint32_t itopk_size, uint32_t search_width, 
                        uint32_t min_iterations, uint32_t max_iterations,
                        uint32_t hash_bitlen) {
        search_params_.itopk_size = itopk_size;
        search_params_.search_width = search_width;
        search_params_.min_iterations = min_iterations;
        search_params_.max_iterations = max_iterations;
        search_params_.hash_bitlen = hash_bitlen;
    };

    void save(const std::string& filepath);
    void load(const std::string& filepath);

    size_t size() const { return current_size_; }

private:
    uint32_t dim_;
    uint32_t graph_degree_;
    size_t current_size_;

    // Host 数据 (Source of Truth)
    std::vector<float> h_data_;
    std::vector<uint32_t> h_graph_;

    // Device VMM 内存管理器
    std::unique_ptr<DeviceBufferVMM> d_data_vmm_;
    std::unique_ptr<DeviceBufferVMM> d_graph_vmm_;

    // 内部参数 (硬编码或构造时确定)
    BuildParams build_params_;
    SearchParams search_params_;  // 用于 Query
    SearchParams insert_params_;  // 用于 Insert 内部的搜索
};

} // namespace cagra