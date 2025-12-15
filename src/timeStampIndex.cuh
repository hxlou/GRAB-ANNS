#pragma once

#include "../src/cagraIndex.hpp" // 引用你修改后的 cagra::CagraIndex
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <mutex>
#include <shared_mutex> // 使用读写锁

namespace timestamp {

// =============================================================================
// 辅助结构体定义
// =============================================================================

// 1. 虚拟节点元数据
// 作用：当我们在虚拟层搜到一个点时，通过这个结构体知道该去查哪个时间桶
struct VirtualPointMeta {
    uint64_t target_timestamp;
    std::vector<uint32_t> entry_points; // 对应桶内的入口点 Local IDs
};

// 2. 桶上下文信息
// 作用：存储该桶的元数据以及 ID 映射表
struct BucketContext {
    std::vector<uint64_t> local_to_global;
    size_t num_dirty;
};

struct VectorLocation {
    uint64_t timestamp;
    size_t local_index;
};

// =============================================================================
// TimeStampIndex 类定义
// =============================================================================
class TimeStampIndex {
public:
    /**
     * @brief 构造函数
     * @param dim 向量维度 (必须与底层 CAGRA 一致)
     * @param degree 图的度数 (默认 32)
     * @param cluster_ratio 聚类比率 (例如 1000，表示每 1000 个向量生成 1 个虚拟导航点)
     */
    TimeStampIndex(uint32_t dim, uint32_t degree = 32, size_t cluster_ratio = 1000);
    
    ~TimeStampIndex();

    uint64_t insert(const float* vectors, const uint64_t* timestamps, size_t num_vectors);

    void build_virtual_index();

    void query(const float* query, size_t topk, 
               int64_t* out_indices, float* out_dists,
               int probe_buckets = 5);

    void query_by_timestamp(const float* query, size_t topk, 
                            uint64_t timestamp,
                            int64_t* out_indices, float* out_dists);

    size_t size() const;

private:
    // 获取或创建桶 (内部非线程安全，需外部加锁)
    // 返回 pair: <Bucket指针, Context指针>
    std::pair<cagra::CagraIndex*, BucketContext*> get_or_create_bucket(uint64_t ts);

    // ==========================================
    // 成员变量
    // ==========================================
    
    uint32_t dim_;
    uint32_t graph_degree_;
    size_t cluster_ratio_;
    
    uint64_t global_count_ = 0; // 全局 ID 计数器

    // 读写锁：保护 buckets_ 和 virtual_index_ 的结构安全
    // query 使用 shared_lock，insert/build 使用 unique_lock
    mutable std::shared_mutex mutex_;

    // --- 数据层 (L1) ---
    // 使用 map 保持时间有序
    std::map<uint64_t, std::unique_ptr<cagra::CagraIndex>> buckets_;
    std::map<uint64_t, BucketContext> bucket_contexts_;

    // --- 导航层 (L2) ---
    // 存储所有桶的聚类中心
    std::unique_ptr<cagra::CagraIndex> virtual_index_;
    std::vector<VirtualPointMeta> virtual_meta_; 

    std::vector<VectorLocation> id_to_location_; // 全局 ID 到 位置 的映射
};

} // namespace timestamp