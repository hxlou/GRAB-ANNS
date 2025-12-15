#include "timeStampIndex.cuh"
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>

// FAISS 头文件
#include <faiss/Clustering.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/utils/random.h>

namespace timestamp {

// =============================================================================
// 构造与析构
// =============================================================================

TimeStampIndex::TimeStampIndex(uint32_t dim, uint32_t degree, size_t cluster_ratio)
    : dim_(dim), 
      graph_degree_(degree), 
      cluster_ratio_(cluster_ratio),
      global_count_(0)
{
    // 可以在这里预留一些空间，例如 id_to_location_
    // id_to_location_.reserve(100000); 
}

TimeStampIndex::~TimeStampIndex() {
    std::cout << "[TimeStampIndex] Destroyed. Total vectors managed: " << global_count_ << std::endl;
}

// =============================================================================
// 内部辅助函数
// =============================================================================
std::pair<cagra::CagraIndex*, BucketContext*> TimeStampIndex::get_or_create_bucket(uint64_t ts) {
    auto it = buckets_.find(ts);
    
    if (it == buckets_.end()) {
        // 1. 创建新的 CagraIndex 桶
        auto new_index = std::make_unique<cagra::CagraIndex>(dim_, graph_degree_);
        cagra::CagraIndex* ptr = new_index.get();
        buckets_[ts] = std::move(new_index);
        
        // 2. 初始化上下文
        BucketContext ctx;
        ctx.num_dirty = 0;
        ctx.local_to_global.reserve(1024); 
        bucket_contexts_[ts] = std::move(ctx);
        
        return {ptr, &bucket_contexts_[ts]};
    }
    
    return {it->second.get(), &bucket_contexts_[ts]};
}

size_t TimeStampIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return global_count_;
}

// =============================================================================
// 核心逻辑：插入
// =============================================================================
uint64_t TimeStampIndex::insert(const float* vectors, const uint64_t* timestamps, size_t num_vectors) {
    // 1. 获取写锁 (互斥)
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // 2. 确定这批数据的起始 Global ID
    uint64_t start_global_id = global_count_;
    
    // 更新总数并扩容映射表
    global_count_ += num_vectors;
    if (id_to_location_.size() < global_count_) {
        id_to_location_.resize(global_count_);
    }

    // 3. 数据分组 (Grouping)
    // 为了批量插入到底层 CAGRA，我们需要将散乱的数据按时间戳归类
    // 结构：TimeStamp -> { Vectors, GlobalIDs }
    struct BatchGroup {
        std::vector<float> vecs;
        std::vector<uint64_t> gids;
    };
    
    // 使用 map 自动排序，虽然这里顺序不重要，但 map 比 unordered_map 在这种场景下更省内存且稳定
    std::map<uint64_t, BatchGroup> groups;

    for (size_t i = 0; i < num_vectors; ++i) {
        uint64_t ts = timestamps[i];
        uint64_t gid = start_global_id + i;
        
        // 获取当前向量的指针
        const float* vec_ptr = vectors + i * dim_;
        
        // 拷贝数据到临时 buffer
        // (注：这会有一次 Host 端内存拷贝，但为了获得连续内存传给底层是必须的)
        BatchGroup& group = groups[ts];
        group.vecs.insert(group.vecs.end(), vec_ptr, vec_ptr + dim_);
        group.gids.push_back(gid);
    }

    // 4. 分发到各个 Bucket 并维护元数据
    for (auto& [ts, group] : groups) {
        // A. 获取桶
        auto [bucket, context] = get_or_create_bucket(ts);
        
        // B. 获取插入前的桶大小 (作为这批数据的起始 Local ID)
        // 注意：CagraIndex::size() 应该返回当前已有的向量数
        size_t start_local_index = bucket->size();
        size_t count = group.gids.size();

        // C. 插入到底层索引
        // 调用 CagraIndex::insert 进行增量构建
        // 这一步是耗时的，因为它涉及 GPU 计算
        std::cout << "[TimeStampIndex] Inserting " << count 
                  << " vectors into bucket ts=" << ts 
                  << " (local start index=" << start_local_index << ")" << std::endl;
        bucket->insert(count, group.vecs.data());

        // D. 维护 ID 映射 (Metadata Update)
        for (size_t k = 0; k < count; ++k) {
            uint64_t gid = group.gids[k];
            size_t lid = start_local_index + k;

            // 1. Global -> Local 映射
            id_to_location_[gid] = VectorLocation{ts, lid};

            // 2. Local -> Global 映射
            context->local_to_global.push_back(gid);
        }

        // E. 标记 Dirty (用于触发后续的虚拟节点更新)
        context->num_dirty += count;
    }

    // 返回这批数据的起始 ID 给用户
    return start_global_id;
}

void TimeStampIndex::build_virtual_index() {
    std::unique_lock<std::shared_mutex> lock(mutex_); // 加写锁
    
    if (buckets_.empty()) return;
    
    std::cout << "[TimeStampIndex] Rebuilding virtual index..." << std::endl;

    std::vector<float> all_centroids;
    std::vector<VirtualPointMeta> new_virtual_meta;

    faiss::gpu::StandardGpuResources gpu_res;
    gpu_res.setTempMemory(512 * 1024 * 1024);

    std::cout << "[TimeStampIndex] Processing " << buckets_.size() << " buckets." << std::endl;
    for (auto& [ts, bucket] : buckets_) {
        // 1. 获取数据
        const float* h_data = (const float*)bucket->get_data(); // 确保类型匹配
        size_t num_vectors = bucket->size();

        if (num_vectors == 0) continue;

        // K = sqrt(N)
        int num_clusters = 16 * std::max(1, (int)std::sqrt(num_vectors));
        
        // 2. K-Means 聚类
        std::cout << "start clustering bucket ts=" << ts 
                  << " num_vectors=" << num_vectors 
                  << " num_clusters=" << num_clusters << std::endl;
        faiss::Clustering clus(dim_, num_clusters); // 构造时传入 dim 和 k
        clus.niter = 30;
        clus.verbose = false;
        clus.min_points_per_centroid = 1;
        
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0; // 建议用 device 0 或参数控制
        config.useFloat16 = false; // 建议 float32 保证精度

        faiss::gpu::GpuIndexFlatL2 index(&gpu_res, dim_, config);
        clus.train(num_vectors, h_data, index);

        std::cout << "clustering done." << std::endl;

        // 3. 在桶内搜索入口点 (使用 centroids 作为 query)
        int entry_num_per_centroid = 5; // 每个聚类中心找 5 个入口点
        
        // 【关键修正】必须使用 int64_t 接收 Cagra 的输出
        std::vector<int64_t> raw_entry_indices(num_clusters * entry_num_per_centroid);
        std::vector<float> raw_entry_dists(num_clusters * entry_num_per_centroid);

        bucket->query(
            clus.centroids.data(), 
            num_clusters, 
            entry_num_per_centroid,
            raw_entry_indices.data(),
            raw_entry_dists.data()
            // seeds = nullptr (这里不需要 seed，这是在找 seed)
        );

        // 4. 汇总结果
        for (int i = 0; i < num_clusters; ++i) {
            // 保存聚类中心向量
            all_centroids.insert(
                all_centroids.end(),
                clus.centroids.begin() + i * dim_,
                clus.centroids.begin() + (i + 1) * dim_
            );

            // 保存元数据
            VirtualPointMeta vpm;
            vpm.target_timestamp = ts;
            
            // 将 int64 转为 uint32 存入元数据
            for (int k = 0; k < entry_num_per_centroid; ++k) {
                int64_t idx = raw_entry_indices[i * entry_num_per_centroid + k];
                if (idx >= 0) { // 过滤无效索引
                    vpm.entry_points.push_back(static_cast<uint32_t>(idx));
                }
            }
            new_virtual_meta.push_back(std::move(vpm));
        }
    }

    // 5. 构建顶层虚拟索引
    if (all_centroids.empty()) {
        virtual_index_.reset();
        virtual_meta_.clear();
        return;
    }

    // 重建
    virtual_index_ = std::make_unique<cagra::CagraIndex>(dim_, graph_degree_);
    virtual_index_->add(all_centroids.size() / dim_, all_centroids.data());
    virtual_index_->build();

    virtual_meta_ = std::move(new_virtual_meta);

    std::cout << "[TimeStampIndex] Rebuilt done. Nodes: " << virtual_meta_.size() << std::endl;
}

void TimeStampIndex::query(const float* query, size_t topk, 
                           int64_t* out_indices, float* out_dists,
                           int probe_buckets) 
{
    // 1. 获取读锁
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // 0. 边界检查
    if (!virtual_index_ || buckets_.empty()) {
        for(size_t i=0; i<topk; ++i) { out_indices[i] = -1; out_dists[i] = -1.0f; }
        return;
    }

    // -------------------------------------------------------
    // Phase 1: L1 Search (Virtual Layer) - 扩大搜索范围
    // -------------------------------------------------------
    
    // 策略：搜索 5 倍的 probe_buckets，作为投票池
    int virtual_search_k = probe_buckets * 10;
    
    // 限制一下，别超过虚拟点总数
    if (virtual_search_k > virtual_meta_.size()) {
        virtual_search_k = virtual_meta_.size();
    }

    std::vector<int64_t> v_indices(virtual_search_k);
    std::vector<float> v_dists(virtual_search_k);
    
    virtual_index_->query(query, 1, virtual_search_k, v_indices.data(), v_dists.data());

    // -------------------------------------------------------
    // Phase 2: Voting & Routing (投票与路由)
    // -------------------------------------------------------
    
    // 临时结构：统计每个桶的命中次数，并收集种子
    struct BucketCandidate {
        int vote_count = 0;
        std::vector<uint32_t> seeds;
    };
    std::map<uint64_t, BucketCandidate> bucket_votes;

    // 遍历虚拟层搜索结果，进行投票
    for (int i = 0; i < virtual_search_k; ++i) {
        int64_t v_idx = v_indices[i];
        if (v_idx >= 0 && v_idx < virtual_meta_.size()) {
            const auto& meta = virtual_meta_[v_idx];
            
            auto& candidate = bucket_votes[meta.target_timestamp];
            candidate.vote_count++;
            // 收集该虚拟点提供的入口点
            candidate.seeds.insert(candidate.seeds.end(), 
                                   meta.entry_points.begin(), 
                                   meta.entry_points.end());
        }
    }

    // 将 map 转为 vector 进行排序
    struct SortedBucket {
        uint64_t ts;
        int count;
        const std::vector<uint32_t>* seeds_ptr; // 借用指针，避免拷贝
    };
    std::vector<SortedBucket> ranked_buckets;
    ranked_buckets.reserve(bucket_votes.size());

    for (auto& kv : bucket_votes) {
        ranked_buckets.push_back({kv.first, kv.second.vote_count, &kv.second.seeds});
    }

    // 排序：命中次数多的优先 (降序)
    std::sort(ranked_buckets.begin(), ranked_buckets.end(), 
        [](const SortedBucket& a, const SortedBucket& b) {
            return a.count > b.count; 
        }
    );

    // -------------------------------------------------------
    // Phase 3: Task Generation (选择 Top-N 桶)
    // -------------------------------------------------------
    
    // 决定实际要搜的桶数量：min(请求数, 实际命中的桶数)
    int final_buckets_num = std::min((size_t)probe_buckets, ranked_buckets.size());

    struct SearchTask {
        uint64_t ts;
        const std::vector<uint32_t>* seeds;
    };
    std::vector<SearchTask> tasks;
    tasks.reserve(final_buckets_num);

    for (int i = 0; i < final_buckets_num; ++i) {
        tasks.push_back({ranked_buckets[i].ts, ranked_buckets[i].seeds_ptr});
    }

    // -------------------------------------------------------
    // Phase 4: L2 Search (Parallel Execution) - 逻辑不变
    // -------------------------------------------------------
    
    std::vector<std::pair<float, int64_t>> all_candidates;

    #pragma omp parallel
    {
        std::vector<std::pair<float, int64_t>> local_candidates;
        // 如果 topk 很小，预留稍微多一点防止频繁分配
        local_candidates.reserve(tasks.size() * topk + 64); 

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < tasks.size(); ++i) {
            uint64_t ts = tasks[i].ts;
            const auto* seeds_ptr = tasks[i].seeds;

            auto it_bucket = buckets_.find(ts);
            auto it_ctx = bucket_contexts_.find(ts);

            if (it_bucket != buckets_.end() && it_ctx != bucket_contexts_.end()) {
                cagra::CagraIndex* bucket = it_bucket->second.get();
                const BucketContext& ctx = it_ctx->second;

                std::vector<int64_t> bucket_indices(topk);
                std::vector<float> bucket_dists(topk);

                // 执行带 Seed 的查询
                bucket->query(
                    query, 
                    1, 
                    topk, 
                    bucket_indices.data(), 
                    bucket_dists.data(),
                    (uint32_t*)seeds_ptr->data(), 
                    seeds_ptr->size()
                );

                for (size_t k = 0; k < topk; ++k) {
                    int64_t lid = bucket_indices[k];
                    float dist = bucket_dists[k];

                    if (lid >= 0 && lid < ctx.local_to_global.size()) {
                        uint64_t gid = ctx.local_to_global[lid];
                        local_candidates.push_back({dist, (int64_t)gid});
                    }
                }
            }
        }

        #pragma omp critical
        {
            all_candidates.insert(all_candidates.end(), 
                                  local_candidates.begin(), 
                                  local_candidates.end());
        }
    }

    // -------------------------------------------------------
    // Phase 5: Merge & Sort
    // -------------------------------------------------------
    
    if (all_candidates.size() > topk) {
        std::partial_sort(all_candidates.begin(), 
                          all_candidates.begin() + topk, 
                          all_candidates.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
    } else {
        std::sort(all_candidates.begin(), all_candidates.end(), 
            [](const auto& a, const auto& b) {
                return a.first < b.first; 
            });
    }

    for (size_t k = 0; k < topk; ++k) {
        if (k < all_candidates.size()) {
            out_dists[k] = all_candidates[k].first;
            out_indices[k] = all_candidates[k].second;
        } else {
            out_dists[k] = -1.0f; 
            out_indices[k] = -1;
        }
    }
}

} // namespace timestamp