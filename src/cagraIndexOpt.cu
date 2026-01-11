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

CagraIndexOpt::CagraIndexOpt(uint32_t dim, uint32_t graph_degree, uint32_t local_degree, size_t vmm_max_bytes)
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
    local_degree_ = local_degree;
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
        h_timestamps_.data(),
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
        dim_,
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
    uint32_t seeds_per_query = std::max(32u, search_params_.itopk_size); // 默认每个查询给 32 个种子

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
        dim_,
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

// =============================================================================
// 核心：范围查询 (Query Range)
// =============================================================================

void CagraIndexOpt::query_range(const float* host_queries, 
                                size_t num_queries, 
                                int k, 
                                uint64_t start_bucket,  // [start, end)
                                uint64_t end_bucket,
                                int64_t* host_indices, 
                                float* host_dists,
                                uint32_t local_degree)  // 通常传入 graph_degree_ (32) 以启用 Remote Edge
{
    if (current_size_ == 0) return;
    local_degree = local_degree_;
    // =========================================================
    // 1. 种子采样 (Seed Sampling)
    // 从指定的时间桶范围内 [start, end) 采样种子
    // =========================================================
    std::vector<uint32_t> sampled_seeds;
    uint32_t seeds_per_query = std::max(32u, search_params_.itopk_size / 4);

    {        
        // 快速定位范围
        auto it_start = ts_to_ids_.lower_bound(start_bucket);
        auto it_end = ts_to_ids_.lower_bound(end_bucket);

        // 收集候选池
        uint32_t candidate_per_bucket = 500;
        std::vector<uint32_t> candidates;
        candidates.reserve((end_bucket - start_bucket) * candidate_per_bucket);

        // 遍历范围内的桶
        for (auto it = it_start; it != it_end; ++it) {
            const auto& ids = it->second;
            // 简单策略：每个桶取一点，或者全取
            // 避免拷贝太多，只取每个桶的前 100 个
            size_t take = std::min(ids.size(), (size_t)candidate_per_bucket);
            candidates.insert(candidates.end(), ids.begin(), ids.begin() + take);
        }

        if (!candidates.empty()) {
            std::mt19937 rng(std::random_device{}());
            if (candidates.size() <= seeds_per_query) {
                sampled_seeds = candidates;
            } else {
                sampled_seeds.reserve(seeds_per_query);
                std::sample(candidates.begin(), candidates.end(), 
                            std::back_inserter(sampled_seeds), 
                            seeds_per_query, rng);
            }
        }
    }

    uint32_t actual_seeds_count = sampled_seeds.size();

    // 如果该范围内没有任何数据，无法搜索，返回空
    if (actual_seeds_count == 0) {
        for (size_t i = 0; i < num_queries * k; ++i) {
            host_indices[i] = -1;
            host_dists[i] = -1.0f;
        }
        return;
    }

    // =========================================================
    // 2. 准备显存 (Queries & Seeds)
    // =========================================================
    float* d_queries;
    int64_t* d_indices;
    float* d_dists;
    uint32_t* d_seeds;

    CUDA_CHECK(cudaMalloc(&d_queries, num_queries * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists, num_queries * k * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_queries, host_queries, 
                          num_queries * dim_ * sizeof(float), 
                          cudaMemcpyHostToDevice));

    // 广播 Seeds：所有 Query 使用同一组范围内有效的种子
    std::vector<uint32_t> batch_seeds(num_queries * actual_seeds_count);
    for (size_t i = 0; i < num_queries; ++i) {
        std::memcpy(batch_seeds.data() + i * actual_seeds_count, 
                    sampled_seeds.data(), 
                    actual_seeds_count * sizeof(uint32_t));
    }
    CUDA_CHECK(cudaMalloc(&d_seeds, batch_seeds.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_seeds, batch_seeds.data(), batch_seeds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // =========================================================
    // 3. 调用底层 Wrapper
    // =========================================================
    float* d_dataset = (float*)d_data_vmm_->data();
    uint32_t* d_graph = (uint32_t*)d_graph_vmm_->data();
    uint64_t* d_timestamps = (uint64_t*)d_ts_vmm_->data();

    // 调用 cagra_opt.cu 中的 search_bucket_range
    cagra::search_bucket_range(
        d_dataset,
        dim_,
        current_size_,
        d_graph,
        d_timestamps,       // 传入时间戳数组
        graph_degree_,      // total_degree (32)
        local_degree,       // active_degree (通常也是 32，允许走 Remote)
        d_queries,
        (int64_t)num_queries,
        (int64_t)k,
        start_bucket,
        end_bucket,
        search_params_,
        d_indices,
        d_dists,
        d_seeds,
        (uint32_t)actual_seeds_count
    );

    // =========================================================
    // 4. 结果回传 & 清理
    // =========================================================
    CUDA_CHECK(cudaMemcpy(host_indices, d_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dists, d_dists, num_queries * k * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaFree(d_seeds));
}


// =============================================================================
// 核心：增量插入 (Insert) - 支持 Batch 处理 修改：限制数据都为同一个时间戳
// =============================================================================
void CagraIndexOpt::insert(size_t new_vectors, const float* insert_vectors, const uint64_t* insert_timestamps) {
    if (new_vectors == 0) return;
    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t target_ts = insert_timestamps[0];
    // -------------------------------------------------------
    // 1. Host 数据更新 & VMM 物理扩容
    // -------------------------------------------------------
    size_t old_size = current_size_;
    size_t new_total = old_size + new_vectors;

    // add() 负责: resize h_data, h_timestamps, 更新 ts_to_ids_
    add(new_vectors, insert_vectors, insert_timestamps);
    
    // 更新 VMM
    d_data_vmm_->resize(new_total * dim_ * sizeof(float));
    d_graph_vmm_->resize(new_total * graph_degree_ * sizeof(uint32_t));
    // d_ts_vmm_ 在 add() 中已经 resize 并 update 了

    // 拷贝新向量到 GPU
    float* d_dataset = (float*)d_data_vmm_->data();
    uint32_t* d_graph = (uint32_t*)d_graph_vmm_->data();
    uint64_t* d_timestamps = (uint64_t*)d_ts_vmm_->data();

    CUDA_CHECK(cudaMemcpy(d_dataset + old_size * dim_, insert_vectors, 
                          new_vectors * dim_ * sizeof(float), cudaMemcpyHostToDevice));

    // 更新当前大小
    current_size_ = new_total;

    // -------------------------------------------------------
    // 2. Batch 处理图更新
    // -------------------------------------------------------
    // 设定 Batch Size (例如 1024 或由外部指定，这里先硬编码或作为成员变量)
    size_t batch_size = 32;
    uint32_t num_seeds_per_query = std::max(32u, search_params_.itopk_size);
    const float* d_new = d_dataset + old_size * dim_;

    // std::cout << "[CagraIndexOpt] Inserting " << new_vectors << " vectors (Batch Size: " << batch_size << ")..." << std::endl;

    this->setQueryParams(
        128,  // itopk
        4,    // search_width
        0,    // min_iter
        100,   // max_iter
        13    // hash_bitlen
    );

    for (size_t offset = 0; offset < new_vectors; offset += batch_size) {
        size_t current_batch_size = std::min(batch_size, new_vectors - offset);
        // auto t1 = std::chrono::high_resolution_clock::now();
        // A. 准备当前 Batch 的指针
        // 每 1/5 的 Batch 使用启发式种子采样
        bool use_heruistic = false;
        // if ((offset / batch_size) % ((new_vectors / batch_size) / 5 ) == 0 ) {
        //     printf("Now is at insert batch %zu / %zu, use heuristic seed sampling.\n", offset / batch_size, (new_vectors + batch_size -1) / batch_size);
        //     use_heruistic = true;
        // }

        const float* d_batch_queries = d_new + offset * dim_;
        
        // B. 生成随机种子 (Host -> Device)
        std::vector<uint32_t> batch_seeds_host(current_batch_size * num_seeds_per_query);
        
        std::fill(batch_seeds_host.begin(), batch_seeds_host.end(), 0xFFFFFFFF);

        {
            // 采样 ts_to_ids_ 中 target_ts 对应的 ID 作为种子
            auto it = ts_to_ids_.find(target_ts);
            std::vector<uint32_t> candidate_ids;
            if (it != ts_to_ids_.end() && !it->second.empty()) {
                // 由于图更新的比较慢，所以不应该使用这个，应该用old size之前的数据
                candidate_ids = it->second;
                int idx = 0;
                for (; idx < candidate_ids.size(); ++idx) {
                    if (candidate_ids[idx] >= old_size) {
                        break;
                    }
                }
                candidate_ids.resize(idx);
            }
            // 为第一个新插入的节点采样种子
            std::mt19937 rng(std::random_device{}());
            if (candidate_ids.size() <= num_seeds_per_query) {
                std::memcpy(batch_seeds_host.data(), 
                            candidate_ids.data(), 
                            candidate_ids.size() * sizeof(uint32_t));
            } else {
                std::sample(candidate_ids.begin(), candidate_ids.end(), 
                            batch_seeds_host.data(), 
                            num_seeds_per_query, rng);
            }

            // 把第一个节点的种子广播给当前 Batch 的其他节点
            for (size_t i = 1; i < current_batch_size; ++i) {
                std::memcpy(batch_seeds_host.data() + i * num_seeds_per_query, 
                            batch_seeds_host.data(), 
                            num_seeds_per_query * sizeof(uint32_t));
            }

            // // 输出一下所有种子在一行
            // std::cout << "   [Insert Batch " << (offset / batch_size) << "] old size is " << old_size << " Seeds: ";
            // for (size_t s = 0; s < num_seeds_per_query; ++s) {
            //     std::cout << batch_seeds_host[s] << " ";
            // }
            // std::cout << std::endl;
        }

        // 拷贝种子到 GPU (临时显存)
        uint32_t* d_batch_seeds = nullptr;
        size_t seeds_bytes = batch_seeds_host.size() * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_batch_seeds, seeds_bytes));
        CUDA_CHECK(cudaMemcpy(d_batch_seeds, batch_seeds_host.data(), seeds_bytes, cudaMemcpyHostToDevice));

        // C. 调用无状态算法
        // 注意：num_existing 始终传 old_size。因为新插入的节点还未建立反向连接，不适合作为 Search Target。
        // 虽然物理上它们在 d_dataset 里，但 search_opt 内部会根据 num_existing 限制搜索范围。
        // ！！！！！！！！这里手动设置了search参数！！！！！！！！

        // auto t2 = std::chrono::high_resolution_clock::now();

        cagra::insert(
            d_dataset,
            d_graph,
            d_timestamps,
            d_batch_seeds,
            old_size,              // Search Space Limit (只搜老数据)
            current_batch_size,    // Batch Size
            use_heruistic,         // 是否使用启发式种子采样
            target_ts,            // 当前插入数据的时间戳
            d_batch_queries,       // Query Ptr
            dim_,
            graph_degree_,         // 32
            local_degree_,     // 28 (Local)
            search_params_,
            num_seeds_per_query
        );

        // 更新old_size
        old_size += current_batch_size;

        // 清理临时种子
        CUDA_CHECK(cudaFree(d_batch_seeds));
        auto t3 = std::chrono::high_resolution_clock::now();

        // std::chrono::duration<double> prepare_elapsed = t2 - t1;
        // std::chrono::duration<double> insert_elapsed = t3 - t2;
        // std::cout << "   [Batch " << (offset / batch_size) << "] Prepare Time: " << prepare_elapsed.count() << " s, Insert Time: " << insert_elapsed.count() << " s." << std::endl;
        
        // 进度条
        // if ((offset / batch_size) % 500 == 0) std::cout << "now is at batch " << (offset / batch_size)  << " / " <<  (new_vectors + batch_size - 1) / batch_size << std::endl;
    }
    // std::cout << " Done." << std::endl;

    // 同步一下d_graph到h_graph_，保持一致性
    h_graph_.resize(new_total * graph_degree_);
    CUDA_CHECK(cudaMemcpy(h_graph_.data(), d_graph, 
                          new_total * graph_degree_ * sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost));

    auto t2 =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    auto IPS = new_vectors / elapsed.count();
    std::cout << "[CagraIndexOpt] Inserted " << new_vectors << " vectors in " << elapsed.count() << " seconds. (IPS: " << IPS << ")" << std::endl;
}

// =============================================================================
// 序列化: Save
// =============================================================================
void CagraIndexOpt::save(const std::string& filepath) {
    std::cout << "[CagraIndexOpt] Saving index to " << filepath << "..." << std::endl;

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // ============================================================
    // 1. 写入文件头 (Header)
    // ============================================================
    // Magic Number (4 bytes): "MCAG" (Multi-bucket CAGRA)
    const char magic[4] = {'M', 'C', 'A', 'G'};
    ofs.write(magic, 4);

    // Version (4 bytes): 版本号，用于后续格式升级
    uint32_t version = 1;
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

    // ============================================================
    // 2. 写入元数据 (Metadata)
    // ============================================================
    ofs.write(reinterpret_cast<const char*>(&dim_), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(&graph_degree_), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(&local_degree_), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&current_size_), sizeof(size_t));

    // ============================================================
    // 3. 写入时间戳数据 (Timestamps)
    // ============================================================
    // 3.1 正向索引: h_timestamps_ [N]
    size_t ts_size = h_timestamps_.size();
    ofs.write(reinterpret_cast<const char*>(&ts_size), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(h_timestamps_.data()), ts_size * sizeof(uint64_t));

    // 3.2 倒排索引: ts_to_ids_ (map<uint64_t, vector<uint32_t>>)
    size_t num_buckets = ts_to_ids_.size();
    ofs.write(reinterpret_cast<const char*>(&num_buckets), sizeof(size_t));
    for (const auto& entry : ts_to_ids_) {
        uint64_t ts = entry.first;
        const std::vector<uint32_t>& ids = entry.second;
        size_t ids_size = ids.size();

        ofs.write(reinterpret_cast<const char*>(&ts), sizeof(uint64_t));
        ofs.write(reinterpret_cast<const char*>(&ids_size), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(ids.data()), ids_size * sizeof(uint32_t));
    }

    // ============================================================
    // 4. 写入向量数据 (Dataset)
    // ============================================================
    size_t data_size = h_data_.size();
    ofs.write(reinterpret_cast<const char*>(&data_size), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(h_data_.data()), data_size * sizeof(float));

    // ============================================================
    // 5. 写入图数据 (Graph)
    // ============================================================
    size_t graph_size = h_graph_.size();
    ofs.write(reinterpret_cast<const char*>(&graph_size), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(h_graph_.data()), graph_size * sizeof(uint32_t));

    ofs.close();

    std::cout << "[CagraIndexOpt] Save complete. Saved " << current_size_
              << " vectors with " << num_buckets << " buckets." << std::endl;
}

// =============================================================================
// 反序列化: Load
// =============================================================================
void CagraIndexOpt::load(const std::string& filepath) {
    std::cout << "[CagraIndexOpt] Loading index from " << filepath << "..." << std::endl;

    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    // ============================================================
    // 1. 读取并验证文件头
    // ============================================================
    char magic[4];
    ifs.read(magic, 4);
    if (magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'A' || magic[3] != 'G') {
        throw std::runtime_error("Invalid file format: Magic number mismatch");
    }

    uint32_t version;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != 1) {
        throw std::runtime_error("Unsupported version: " + std::to_string(version));
    }

    // ============================================================
    // 2. 读取元数据
    // ============================================================
    ifs.read(reinterpret_cast<char*>(&dim_), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&graph_degree_), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&local_degree_), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&current_size_), sizeof(size_t));

    std::cout << "[CagraIndexOpt] Loading metadata: dim=" << dim_
              << ", graph_degree=" << graph_degree_
              << ", local_degree=" << local_degree_
              << ", size=" << current_size_ << std::endl;

    // ============================================================
    // 3. 读取时间戳数据
    // ============================================================
    // 3.1 正向索引
    size_t ts_size;
    ifs.read(reinterpret_cast<char*>(&ts_size), sizeof(size_t));
    h_timestamps_.resize(ts_size);
    ifs.read(reinterpret_cast<char*>(h_timestamps_.data()), ts_size * sizeof(uint64_t));

    // 3.2 倒排索引
    size_t num_buckets;
    ifs.read(reinterpret_cast<char*>(&num_buckets), sizeof(size_t));
    ts_to_ids_.clear();
    for (size_t i = 0; i < num_buckets; ++i) {
        uint64_t ts;
        size_t ids_size;
        ifs.read(reinterpret_cast<char*>(&ts), sizeof(uint64_t));
        ifs.read(reinterpret_cast<char*>(&ids_size), sizeof(size_t));

        std::vector<uint32_t> ids(ids_size);
        ifs.read(reinterpret_cast<char*>(ids.data()), ids_size * sizeof(uint32_t));
        ts_to_ids_[ts] = std::move(ids);
    }

    // ============================================================
    // 4. 读取向量数据
    // ============================================================
    size_t data_size;
    ifs.read(reinterpret_cast<char*>(&data_size), sizeof(size_t));
    h_data_.resize(data_size);
    ifs.read(reinterpret_cast<char*>(h_data_.data()), data_size * sizeof(float));

    // ============================================================
    // 5. 读取图数据
    // ============================================================
    size_t graph_size;
    ifs.read(reinterpret_cast<char*>(&graph_size), sizeof(size_t));
    h_graph_.resize(graph_size);
    ifs.read(reinterpret_cast<char*>(h_graph_.data()), graph_size * sizeof(uint32_t));

    ifs.close();

    // ============================================================
    // 6. 同步数据到 GPU VMM
    // ============================================================
    // 恢复 VMM 中的数据
    d_data_vmm_->resize(current_size_ * dim_ * sizeof(float));
    float* d_dataset = (float*)d_data_vmm_->data();
    CUDA_CHECK(cudaMemcpy(d_dataset, h_data_.data(),
                          current_size_ * dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    d_ts_vmm_->resize(current_size_ * sizeof(uint64_t));
    uint64_t* d_timestamps = (uint64_t*)d_ts_vmm_->data();
    CUDA_CHECK(cudaMemcpy(d_timestamps, h_timestamps_.data(),
                          current_size_ * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    d_graph_vmm_->resize(current_size_ * graph_degree_ * sizeof(uint32_t));
    uint32_t* d_graph = (uint32_t*)d_graph_vmm_->data();
    CUDA_CHECK(cudaMemcpy(d_graph, h_graph_.data(),
                          current_size_ * graph_degree_ * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    std::cout << "[CagraIndexOpt] Load complete. Loaded " << current_size_
              << " vectors with " << num_buckets << " buckets." << std::endl;
}

} // namespace cagra