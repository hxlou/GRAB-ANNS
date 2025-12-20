#include "cagraIndexOpt.cuh"
#include "cagra_opt.cuh"
#include "cagra.cuh"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <random>

namespace cagra {

// =============================================================================
// 构造与析构
// =============================================================================

CagraIndexOpt::CagraIndexOpt(uint32_t dim, uint32_t graph_degree, size_t vmm_max_bytes)
    : dim_(dim), 
      graph_degree_(graph_degree), 
      current_size_(0),
      remote_edge_rate_(0.5) // 默认 20% 的边用于 Remote Edge (即 32 * 0.2 ~= 6)
{

    size_t data_cap = vmm_max_bytes * 0.70;
    size_t graph_cap = vmm_max_bytes * 0.20;
    size_t ts_cap = vmm_max_bytes * 0.10;

    d_data_vmm_ = std::make_unique<DeviceBufferVMM>(data_cap);
    d_graph_vmm_ = std::make_unique<DeviceBufferVMM>(graph_cap);
    d_ts_vmm_ = std::make_unique<DeviceBufferVMM>(ts_cap);

    // 参数默认值
    build_params_ = {128, graph_degree}; 
    search_params_.itopk_size = 256;
    search_params_.search_width = 6;
    search_params_.min_iterations = 1;
    search_params_.max_iterations = 50;
    search_params_.hash_bitlen = 13;

    // local
    local_degree_ = graph_degree_ - graph_degree_ * remote_edge_rate_;
}

CagraIndexOpt::~CagraIndexOpt() {
    // 智能指针会自动释放 VMM 资源 (cuMemAddressFree 等)
    // vector 和 map 也会自动析构
    // 无需手动操作
}

// =============================================================================
// 数据管理 (Host 端积累)
// =============================================================================

void CagraIndexOpt::add(size_t num_vectors, const float* add_vectors, const uint64_t* add_timestamps) {
    if (num_vectors == 0) return;

    size_t start_idx = h_data_.size() / dim_;
    size_t new_total = start_idx + num_vectors;

    // 1. 追加向量数据 (Host Vector)
    // resize 可能会触发 realloc，数据量大时耗时，建议预留 reserve
    h_data_.resize(new_total * dim_);
    std::memcpy(h_data_.data() + start_idx * dim_, add_vectors, num_vectors * dim_ * sizeof(float));

    // 2. 追加时间戳 (Host Vector)
    // 这是正向索引: ID -> Timestamp
    h_timestamps_.resize(new_total);
    std::memcpy(h_timestamps_.data() + start_idx, add_timestamps, num_vectors * sizeof(uint64_t));

    // 3. 更新倒排索引 (Inverted Index): Timestamp -> List[ID]
    // 遍历新增的数据，逐个插入 map
    for (size_t i = 0; i < num_vectors; ++i) {
        uint32_t global_id = start_idx + i;
        uint64_t ts = add_timestamps[i];
        
        ts_to_ids_[ts].push_back(global_id);
    }

    // 更新device的反查表 d_ts_vmm
    size_t new_ts_bytes = new_total * sizeof(uint64_t);
    d_ts_vmm_->resize(new_ts_bytes);
    
    uint64_t* d_ts_ptr = (uint64_t*)d_ts_vmm_->data();
    
    // 仅拷贝新增的时间戳部分
    CUDA_CHECK(cudaMemcpy(d_ts_ptr + start_idx, 
                          add_timestamps, 
                          num_vectors * sizeof(uint64_t), 
                          cudaMemcpyHostToDevice));
}

// =============================================================================
// 辅助功能: 种子采样 (Sample Seeds)
// =============================================================================

std::vector<uint32_t> CagraIndexOpt::sample_seeds_by_time(uint64_t min_ts, uint64_t max_ts, size_t num) const {
    std::vector<uint32_t> candidates;
    
    // 利用 map 的有序性快速定位范围
    // lower_bound: >= min_ts
    auto it_start = ts_to_ids_.lower_bound(min_ts);
    // upper_bound: > max_ts
    auto it_end = ts_to_ids_.upper_bound(max_ts);

    // 收集范围内所有的 ID
    // 优化提示：如果范围很大，全收集太慢。可以随机跳着收集，或者每个时间戳只取前几个。
    // 这里做简单全收集再采样
    for (auto it = it_start; it != it_end; ++it) {
        const auto& ids = it->second;
        candidates.insert(candidates.end(), ids.begin(), ids.end());
        
        // 如果候选集太大，提前截断，避免内存爆炸
        if (candidates.size() > num * 100) break; 
    }

    if (candidates.empty()) return {};

    // 随机采样
    std::vector<uint32_t> result;
    result.reserve(num);
    
    // 使用 std::sample (C++17) 或者手动 shuffle
    // 这里简单 shuffle
    // 注意：const 函数里不能修改成员，这里我们用的局部变量，没问题
    // 为了随机性，建议传入或使用 thread_local 的 rng，这里简单起见用静态
    static std::mt19937 rng(1234); 
    
    std::sample(candidates.begin(), candidates.end(), 
                std::back_inserter(result), num, rng);
    
    return result;
}

void CagraIndexOpt::build() {
    if (h_data_.empty()) return;

    size_t num_vectors = h_data_.size() / dim_;
    current_size_ = num_vectors;

    std::cout << "[CagraIndexOpt] Building index for " << num_vectors << " vectors..." << std::endl;

    // -----------------------------------------------------------
    // 1. 同步数据到 GPU VMM (物理层准备)
    // -----------------------------------------------------------
    
    // A. 向量数据
    d_data_vmm_->resize(num_vectors * dim_ * sizeof(float));
    float* d_dataset = (float*)d_data_vmm_->data();
    CUDA_CHECK(cudaMemcpy(d_dataset, h_data_.data(), 
                          num_vectors * dim_ * sizeof(float), 
                          cudaMemcpyHostToDevice));

    // B. 时间戳数据
    d_ts_vmm_->resize(num_vectors * sizeof(uint64_t));
    uint64_t* d_timestamps = (uint64_t*)d_ts_vmm_->data();
    CUDA_CHECK(cudaMemcpy(d_timestamps, h_timestamps_.data(), 
                          num_vectors * sizeof(uint64_t), 
                          cudaMemcpyHostToDevice));

    // C. 图数据 (只分配空间，内容由算法填充)
    d_graph_vmm_->resize(num_vectors * graph_degree_ * sizeof(uint32_t));
    uint32_t* d_graph = (uint32_t*)d_graph_vmm_->data();

    // -----------------------------------------------------------
    // 2. 计算分桶信息 (逻辑层准备)
    // -----------------------------------------------------------
    // 这里的假设是：add() 进来的数据在物理上已经大体按时间聚拢了
    // 我们统计连续的时间戳块大小，作为局部构图的边界
    std::vector<size_t> bucket_sizes;
    if (num_vectors > 0) {
        uint64_t current_ts = h_timestamps_[0];
        size_t count = 0;
        
        for (size_t i = 0; i < num_vectors; ++i) {
            if (h_timestamps_[i] == current_ts) {
                count++;
            } else {
                // 遇到新时间戳，结算上一段
                bucket_sizes.push_back(count);
                current_ts = h_timestamps_[i];
                count = 1;
            }
        }
        bucket_sizes.push_back(count); // 结算最后一段
    }

    std::cout << "   [Analysis] Data partitioned into " << bucket_sizes.size() << " physical buckets." << std::endl;

    // -----------------------------------------------------------
    // 3. 调用无状态算法 (Algorithm Call)
    // -----------------------------------------------------------
    
    // 确定 Local vs Global 比例
    // 策略：固定 4 条边做全局连接 (Remote Edges)
    // 剩余 (32-4)=28 条边做局部连接
    uint32_t local_degree = local_degree_;

    // 调用 cagra_opt.cuh 中的核心构建函数
    cagra::build_time_partitioned_graph(
        d_dataset,
        num_vectors,
        dim_,
        d_graph,
        (uint64_t*)d_ts_vmm_->data(),
        bucket_sizes,
        graph_degree_, // total (e.g. 32)
        local_degree   // local (e.g. 28)
    );

    // -----------------------------------------------------------
    // 4. 后处理
    // -----------------------------------------------------------
    
    // 将构建好的图同步回 Host (用于 save/load 序列化)
    h_graph_.resize(num_vectors * graph_degree_);
    CUDA_CHECK(cudaMemcpy(h_graph_.data(), d_graph, 
                          num_vectors * graph_degree_ * sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost));

    std::cout << "[CagraIndexOpt] Build complete." << std::endl;
}

void CagraIndexOpt::query(const float* host_queries, 
                          size_t num_queries, 
                          int k, 
                          uint64_t min_ts,           // 暂时忽略 (用于后续过滤)
                          uint64_t max_ts,           // 暂时忽略
                          int64_t* host_indices, 
                          float* host_dists,
                          const uint32_t* seeds,
                          size_t num_seeds_per_query)
{
    if (current_size_ == 0) return;

    // 1. 准备 Device 内存
    float* d_queries;
    int64_t* d_indices;
    float* d_dists;
    uint32_t* d_seeds = nullptr;

    // 申请显存
    CUDA_CHECK(cudaMalloc(&d_queries, num_queries * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists, num_queries * k * sizeof(float)));

    // 拷贝查询向量
    CUDA_CHECK(cudaMemcpy(d_queries, host_queries, 
                          num_queries * dim_ * sizeof(float), 
                          cudaMemcpyHostToDevice));

    // 处理种子 (如果有)
    if (seeds != nullptr && num_seeds_per_query > 0) {
        size_t seeds_bytes = num_queries * num_seeds_per_query * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_seeds, seeds_bytes));
        CUDA_CHECK(cudaMemcpy(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice));
    }

    // 2. 获取 VMM 数据指针
    float* d_dataset = (float*)d_data_vmm_->data();
    uint32_t* d_graph = (uint32_t*)d_graph_vmm_->data();

    // 3. 调用底层无状态算法 (search_opt)
    // 这里我们只进行全量搜索，不传 timestamp 相关的参数
    cagra::search_opt(
        d_dataset, 
        current_size_, 
        d_graph, 
        graph_degree_, 
        d_queries, 
        (int64_t)num_queries, 
        (int64_t)k, 
        search_params_, 
        d_indices, 
        d_dists,
        d_seeds,            // 传入 GPU 上的种子指针
        num_seeds_per_query // 每个查询的种子数
    );

    // 4. 拷回结果
    CUDA_CHECK(cudaMemcpy(host_indices, d_indices, 
                          num_queries * k * sizeof(int64_t), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dists, d_dists, 
                          num_queries * k * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    // 5. 清理
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_dists));
    if (d_seeds) CUDA_CHECK(cudaFree(d_seeds));
}

#include "index/cagra_opt.cuh" // 包含 search_bucket_opt

// ...

// =============================================================================
// 核心：局部查询 (Query Local / Bucket)
// =============================================================================

void CagraIndexOpt::query_local(const float* host_queries, 
                                size_t num_queries, 
                                int k, 
                                uint64_t target_timestamp, // 指定桶
                                int64_t* host_indices, 
                                float* host_dists,
                                uint32_t local_degree)
{
    if (current_size_ == 0) return;
    local_degree = local_degree_;
    // 1. 内部采样种子 (Internal Seed Sampling)
    // -----------------------------------------------------
    std::vector<uint32_t> sampled_seeds;
    uint32_t seeds_per_query = std::max(32u, search_params_.itopk_size / 4); // 默认每个查询给 32 个种子

    {
        auto it = ts_to_ids_.find(target_timestamp);
        
        if (it != ts_to_ids_.end() && !it->second.empty()) {
            const auto& bucket_ids = it->second;
            
            // 如果桶内数据极少，有多少用多少
            if (bucket_ids.size() <= seeds_per_query) {
                sampled_seeds = bucket_ids;
            } else {
                // 随机采样
                sampled_seeds.reserve(seeds_per_query);
                // 使用静态随机引擎避免重复构造开销
                std::mt19937 rng(std::random_device{}());
                
                std::sample(bucket_ids.begin(), bucket_ids.end(), 
                            std::back_inserter(sampled_seeds), 
                            seeds_per_query, rng);
            }
        }
    }

    uint32_t actual_seeds_count = sampled_seeds.size();
    
    // 如果该时间戳下没有任何数据，填空结果并返回
    if (actual_seeds_count == 0) {
        // std::cerr << "Warning: No data found for timestamp " << target_timestamp << std::endl;
        for (size_t i = 0; i < num_queries * k; ++i) {
            host_indices[i] = -1;
            host_dists[i] = -1.0f; // 或者 MAX_FLOAT
        }
        return;
    }

    // 2. 准备 Device 内存
    // -----------------------------------------------------
    float* d_queries;
    int64_t* d_indices;
    float* d_dists;
    uint32_t* d_seeds;

    CUDA_CHECK(cudaMalloc(&d_queries, num_queries * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists, num_queries * k * sizeof(float)));

    // 拷贝查询
    CUDA_CHECK(cudaMemcpy(d_queries, host_queries, 
                          num_queries * dim_ * sizeof(float), 
                          cudaMemcpyHostToDevice));

    // 拷贝种子
    // 策略：我们生成了一组种子，广播给所有 Query 使用
    // 我们需要在 GPU 上展开成 [num_queries * actual_seeds_count] 的数组
    // 或者，我们可以只传一份种子到 GPU，但在 Kernel 里所有 Query 读同一块内存？
    // 为了适配现有的 search_bucket_opt 接口（它期望 seed_ptr 对应每个 query 有独立的偏移），
    // 我们需要在 Host 端把种子复制 num_queries 份。
    
    std::vector<uint32_t> batch_seeds(num_queries * actual_seeds_count);
    for (size_t i = 0; i < num_queries; ++i) {
        std::memcpy(batch_seeds.data() + i * actual_seeds_count, 
                    sampled_seeds.data(), 
                    actual_seeds_count * sizeof(uint32_t));
    }

    size_t seeds_bytes = batch_seeds.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_seeds, seeds_bytes));
    CUDA_CHECK(cudaMemcpy(d_seeds, batch_seeds.data(), seeds_bytes, cudaMemcpyHostToDevice));

    // 3. 获取 VMM 指针
    float* d_dataset = (float*)d_data_vmm_->data();
    uint32_t* d_graph = (uint32_t*)d_graph_vmm_->data();

    // 4. 调用底层 search_bucket_opt
    // -----------------------------------------------------
    cagra::search_bucket_opt(
        d_dataset, 
        current_size_, 
        d_graph, 
        graph_degree_,       // total_degree (32)
        local_degree,        // local_degree (28, Active)
        d_queries, 
        (int64_t)num_queries, 
        (int64_t)k, 
        search_params_, 
        d_indices, 
        d_dists,
        d_seeds, 
        (uint32_t)actual_seeds_count
    );

    // 5. 拷回结果
    CUDA_CHECK(cudaMemcpy(host_indices, d_indices, 
                          num_queries * k * sizeof(int64_t), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dists, d_dists, 
                          num_queries * k * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    // 6. 清理
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaFree(d_seeds));
}
} // namespace cagra