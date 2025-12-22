#pragma once

#include "cagra.cuh"          // 包含底层算法 (build, search...)
#include "vmm_allocator.cuh"  // 包含 VMM 显存管理
#include <string>
#include <vector>
#include <map>                      // 用于倒排索引
#include <memory>
#include <shared_mutex>             // 用于线程安全

namespace cagra {

class CagraIndexOpt {
public:
    /**
     * @brief 初始化优化版索引管理器
     * @param dim 向量维度 (固定 1024)
     * @param graph_degree 最终图度数 (默认 32)
     * @param vmm_max_bytes VMM 预留的最大显存空间 (默认 20GB)
     */
    CagraIndexOpt(uint32_t dim, 
                  uint32_t graph_degree = 32,
                  size_t vmm_max_bytes = 20ULL * 1024 * 1024 * 1024);

    ~CagraIndexOpt();

    /**
     * @brief [Phase 1] 数据积累
     * 将向量和时间戳追加到 Host 内存缓存中，并更新 CPU 端的倒排映射表。
     * 
     * @param num_vectors 数据数量
     * @param add_vectors 向量数据指针
     * @param add_timestamps 时间戳数据指针 <--- 新增
     */
    void add(size_t num_vectors, const float* add_vectors, const uint64_t* add_timestamps);

    /**
     * @brief [Phase 2] 全量构建
     * 1. 将 Host 数据 (向量 + 时间戳) 同步到 GPU VMM。
     * 2. 生成 KNN 图。
     * 3. 执行 CAGRA 优化 (Prune -> Reverse -> Merge)。
     * 4. 存储最终图到 VMM。
     */
    void build();

    /**
     * @brief [Phase 3] 增量插入
     * 1. 更新 Host 数据和映射表。
     * 2. 扩容 VMM 并追加新数据 (向量 + 时间戳) 到 GPU。
     * 3. (TODO) 执行增量图更新。
     */
    void insert(size_t new_vectors, const float* insert_vectors, const uint64_t* insert_timestamps);

    /**
     * @brief 向量查询 (支持时间过滤 + Seed 导航)
     * 
     * @param host_queries 查询向量 (Host)
     * @param num_queries 查询数量
     * @param k Top-K
     * @param min_ts 最小时间戳 (过滤条件) <--- 新增
     * @param max_ts 最大时间戳 (过滤条件) <--- 新增
     * @param host_indices 输出 ID (Host)
     * @param host_dists 输出距离 (Host)
     * @param seeds [可选] 外部传入的种子 (Host)
     * @param num_seeds_per_query [可选] 每个查询的种子数
     */
    void query(const float* host_queries, 
               size_t num_queries, 
               int k, 
               uint64_t min_ts,           // Time Filter Start
               uint64_t max_ts,           // Time Filter End
               int64_t* host_indices, 
               float* host_dists,
               const uint32_t* seeds = nullptr,
               size_t num_seeds_per_query = 0);

    void query_local(const float* host_queries, 
                        size_t num_queries, 
                        int k, 
                        uint64_t target_timestamp, // 指定桶
                        int64_t* host_indices, 
                        float* host_dists,
                        uint32_t local_degree);

    void query_range(const float* host_queries, 
                        size_t num_queries, 
                        int k, 
                        uint64_t start_bucket,  // 指定范围 [start, end)
                        uint64_t end_bucket,
                        int64_t* host_indices, 
                        float* host_dists,
                        uint32_t local_degree);

    // --- 参数设置 ---
    void setBuildParams(uint32_t inter_degree, uint32_t graph_degree) {
        build_params_.intermediate_degree = inter_degree;
        build_params_.graph_degree = graph_degree;
    }

    void setQueryParams(uint32_t itopk_size, uint32_t search_width, 
                        uint32_t min_iterations, uint32_t max_iterations,
                        uint32_t hash_bitlen) {
        search_params_.itopk_size = itopk_size;
        search_params_.search_width = search_width;
        search_params_.min_iterations = min_iterations;
        search_params_.max_iterations = max_iterations;
        search_params_.hash_bitlen = hash_bitlen;
    }

    // --- 数据访问 ---
    const float* get_data() const { return h_data_.data(); }
    const uint32_t* get_graph() const { return h_graph_.data(); }
    std::vector<uint32_t> get_ids_by_timestamp(uint64_t ts) const {
        if (ts_to_ids_.count(ts)) return ts_to_ids_.at(ts);
        return {};
    }
    size_t size() const { return current_size_; }

    // 获取特定时间范围内的随机种子 (用于辅助搜索)
    std::vector<uint32_t> sample_seeds_by_time(uint64_t min_ts, uint64_t max_ts, size_t num) const;

    // --- 序列化 ---
    void save(const std::string& filepath);
    void load(const std::string& filepath);

private:
    uint32_t dim_;
    uint32_t graph_degree_;
    size_t current_size_;

    // ==========================================
    // 1. Host 存储 (Source of Truth)
    // ==========================================
    std::vector<float> h_data_;
    std::vector<uint32_t> h_graph_;

    std::unique_ptr<DeviceBufferVMM> d_data_vmm_;
    std::unique_ptr<DeviceBufferVMM> d_graph_vmm_;

    std::map<uint64_t, std::vector<uint32_t>> ts_to_ids_;       // host 端使用，快速根据 ts 找到有哪些数据
    std::vector<uint64_t> h_timestamps_;                        // host 端，快速根据 index 找到属于哪个时间戳
    std::unique_ptr<DeviceBufferVMM> d_ts_vmm_;                 // device 端使用，快速根据 index 找到属于哪个时间戳

    BuildParams build_params_;
    SearchParams search_params_;
    double remote_edge_rate_;                                   // 
    size_t local_degree_;
};

} // namespace cagra