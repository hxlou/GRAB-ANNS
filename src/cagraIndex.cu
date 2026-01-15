#include "cagraIndex.hpp"
#include "cagra_opt.cuh"
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <random>
namespace cagra {

CagraIndex::CagraIndex(uint32_t dim, uint32_t graph_degree, size_t vmm_max_bytes)
    : dim_(dim), graph_degree_(graph_degree), current_size_(0) 
{
    // VMM 初始化
    size_t data_cap = vmm_max_bytes / 2;
    size_t graph_cap = vmm_max_bytes / 2;
    d_data_vmm_ = std::make_unique<DeviceBufferVMM>(data_cap);
    d_graph_vmm_ = std::make_unique<DeviceBufferVMM>(graph_cap);
    d_ts_vmm_ = std::make_unique<DeviceBufferVMM>(10 * 1024 * 1024 * sizeof(uint64_t)); // 时间戳 VMM，初始1M大小


    // 1. 构建参数初始化
    build_params_.graph_degree = graph_degree;
    build_params_.intermediate_degree = std::max(128u, graph_degree * 2);

    // 2. 查询参数初始化 (用于 query 接口)
    search_params_.itopk_size = 256;
    search_params_.search_width = 6; // 标准查询平衡参数
    search_params_.max_iterations = 50;
    search_params_.hash_bitlen = 13;            // 目前最大为13，哈希表大小为 32KB      hash_size = (1ull << hash_bitlen) * sizeof(uint32_t);
                                                // 13 对应最大可以储存 8192 个元素
                                                // 3090ti 的 shared memory默认大小48KB，最大可以手动设置100KB，因此这个参数最大设置为14，对应哈希表64KB

    // 3. 插入参数初始化 (用于 insert 内部搜索，需要更激进以保证连通性)
    insert_params_.itopk_size = 128;
    insert_params_.search_width = 6; // 插入时宽搜，找得更准
    insert_params_.max_iterations = 50;
    insert_params_.hash_bitlen = 12;

    // 初始化辅助信息
    h_timestamps_.reserve(1024 * 1024);
}

CagraIndex::~CagraIndex() = default;

// =========================================================
// 1. add: 纯 Host 数据积累
// =========================================================
void CagraIndex::add(size_t num_vectors, const float* add_vectors, const uint64_t* add_timestamps) {
    if (num_vectors == 0) return;
    
    // 直接追加到 std::vector
    h_data_.insert(h_data_.end(), add_vectors, add_vectors + num_vectors * dim_);
    
    // 更新计数 (此时 graph 还没建，不用管 h_graph_)
    current_size_ = h_data_.size() / dim_;

    // 更新 ts_to_ids
    if (add_timestamps != nullptr) {
        for (size_t i = 0; i < num_vectors; ++i) {
            uint64_t ts = add_timestamps[i];
            h_timestamps_.push_back(ts);
            ts_to_ids_[ts].push_back(current_size_ - num_vectors + i);
        }
    }

    // 同步数据到d_ts_vmm_
    if (add_timestamps != nullptr) {
        printf(">> [CagraIndex::add] Syncing %zu timestamps to Device VMM.\n", h_timestamps_.size());
        size_t ts_bytes = h_timestamps_.size() * sizeof(uint64_t);
        d_ts_vmm_->resize(ts_bytes);
        CUDA_CHECK(cudaMemcpy(d_ts_vmm_->data(), h_timestamps_.data(), ts_bytes, cudaMemcpyHostToDevice));
    }
    
    // std::cout << "[CagraIndex::add] Added " << num_vectors << " vectors. Total: " << current_size_ << std::endl;
}

// =========================================================
// 2. build: 全量构建
// =========================================================
void CagraIndex::build() {
    if (current_size_ == 0) return;
    
    // std::cout << ">> [CagraIndex::build] Starting build for " << current_size_ << " vectors..." << std::endl;

    // 1. 调整 Device VMM 大小以匹配 Host 数据
    size_t data_bytes = current_size_ * dim_ * sizeof(float);
    d_data_vmm_->resize(data_bytes);
    
    // 2. 将积累的 Host 数据一次性同步到 Device
    CUDA_CHECK(cudaMemcpy(d_data_vmm_->data(), h_data_.data(), data_bytes, cudaMemcpyHostToDevice));

    // 3. 生成 KNN (使用内部 build_params_)
    std::vector<uint32_t> h_raw_knn(current_size_ * build_params_.intermediate_degree);
    
    cagra::generate_knn_graph((float*)d_data_vmm_->data(), 
                              current_size_, dim_, 
                              build_params_.intermediate_degree, 
                              h_raw_knn.data());

    uint32_t* d_raw_knn;
    CUDA_CHECK(cudaMalloc(&d_raw_knn, h_raw_knn.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_raw_knn, h_raw_knn.data(), h_raw_knn.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 4. 构建 CAGRA 图
    size_t graph_bytes = current_size_ * graph_degree_ * sizeof(uint32_t);
    d_graph_vmm_->resize(graph_bytes);

    uint32_t* d_constructed_graph = nullptr;
    
    // 调用底层 build
    cagra::build((float*)d_data_vmm_->data(), current_size_, d_raw_knn, build_params_, &d_constructed_graph);

    // 5. 拷贝回 VMM 和 Host
    CUDA_CHECK(cudaMemcpy(d_graph_vmm_->data(), d_constructed_graph, graph_bytes, cudaMemcpyDeviceToDevice));
    
    // 更新 Host 端的图镜像
    h_graph_.resize(current_size_ * graph_degree_);
    CUDA_CHECK(cudaMemcpy(h_graph_.data(), d_constructed_graph, graph_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_raw_knn));
    CUDA_CHECK(cudaFree(d_constructed_graph));

    // std::cout << ">> [CagraIndex::build] Finished." << std::endl;
}

// =========================================================
// 3. insert: 增量插入
// =========================================================
void CagraIndex::insert(size_t new_vectors, const float* insert_vectors, const uint64_t* insert_timestamps) {
    if (new_vectors == 0) return;

    std::cout << ">> [CagraIndex::insert] Preparing to insert " << new_vectors << " vectors." << std::endl;
    // 如果是一个全新的图，直接调用 build
    if (current_size_ == 0) {
        // std::cout << ">> [CagraIndex::insert] Index is empty, performing build instead." << std::endl;
        add(new_vectors, insert_vectors, insert_timestamps);
        build();
        return;
    }
    std::cout << ">> [CagraIndex::insert] Inserting into existing index of size " << current_size_ << "." << std::endl;

    size_t old_size = current_size_;
    size_t new_size = old_size + new_vectors;

    std::cout << ">> [CagraIndex::insert] Inserting " << new_vectors << " vectors..." << std::endl;

    // 1. 扩展 Host 数据
    h_data_.insert(h_data_.end(), insert_vectors, insert_vectors + new_vectors * dim_);
    // Graph 扩展占位 (初始化为 INVALID)
    h_graph_.resize(new_size * graph_degree_, 0xFFFFFFFF);

    // 2. 扩展 Device VMM
    size_t new_data_bytes = new_size * dim_ * sizeof(float);
    size_t new_graph_bytes = new_size * graph_degree_ * sizeof(uint32_t);
    
    d_data_vmm_->resize(new_data_bytes);
    d_graph_vmm_->resize(new_graph_bytes);

    // 3. 拷贝新数据到 Device (追加到末尾)
    float* d_new_data_ptr = (float*)d_data_vmm_->data() + old_size * dim_;
    CUDA_CHECK(cudaMemcpy(d_new_data_ptr, insert_vectors, new_vectors * dim_ * sizeof(float), cudaMemcpyHostToDevice));

    // 4. 执行增量插入逻辑
    // 这里的插入，我们每次插入一个batch的大小
    int batch = 64;
    for (size_t offset = 0; offset < new_vectors; offset += batch) {
        if (offset % 2048 == 0) {
            std::cout << "    - Inserting vector " << offset << " / " << new_vectors << std::endl;
        }
        size_t current_batch = std::min(batch, static_cast<int>(new_vectors - offset));
        cagra::insert((float*)d_data_vmm_->data(),
                      old_size + offset,
                      current_batch,
                      d_new_data_ptr + offset * dim_,
                      (uint32_t*)d_graph_vmm_->data(),
                      h_graph_.data(), // 传入完整的 Host 图供 CPU 更新
                      graph_degree_,
                      insert_params_);
    }

    // 5. 更新状态
    current_size_ = new_size;
}

// =========================================================
// 4. query
// =========================================================
void CagraIndex::query(const float* host_queries, 
               size_t num_queries, 
               int k,
               int64_t* host_indices, 
               float* host_dists,
               uint32_t* seeds,
               size_t num_seeds_per_query)
{
    if (current_size_ == 0) return;

    // 使用内部 search_params_，不暴露给外部
    
    float* d_queries;
    CUDA_CHECK(cudaMalloc(&d_queries, num_queries * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_queries, host_queries, num_queries * dim_ * sizeof(float), cudaMemcpyHostToDevice));

    int64_t* d_indices;
    float* d_dists;
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists, num_queries * k * sizeof(float)));

    uint32_t* d_seed_ptr = nullptr;
    if (seeds != nullptr && num_seeds_per_query > 0) {
        CUDA_CHECK(cudaMalloc(&d_seed_ptr, num_queries * num_seeds_per_query * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_seed_ptr, seeds, num_queries * num_seeds_per_query * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    cagra::search((float*)d_data_vmm_->data(), 
                  current_size_, 
                  this->dim_,
                  (uint32_t*)d_graph_vmm_->data(), 
                  graph_degree_, 
                  d_queries, 
                  num_queries, 
                  k, 
                  search_params_, // 使用成员变量
                  d_indices, 
                  d_dists,
                  d_seed_ptr,
                  num_seeds_per_query);

    CUDA_CHECK(cudaMemcpy(host_indices, d_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dists, d_dists, num_queries * k * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_dists));
}

void CagraIndex::query_range(const float* host_queries,
                     size_t num_queries,
                     int k,
                     float range_radius,
                     uint64_t start_ts,
                     uint64_t end_ts,
                     int64_t* host_indices,
                     float* host_dists,
                     uint32_t* seeds,
                     size_t num_seeds_per_query)
{
    if (current_size_ == 0) return;

    // 使用内部 search_params_，不暴露给外部
    
    float* d_queries;
    CUDA_CHECK(cudaMalloc(&d_queries, num_queries * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_queries, host_queries, num_queries * dim_ * sizeof(float), cudaMemcpyHostToDevice));

    int64_t* d_indices;
    float* d_dists;
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * k * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dists, num_queries * k * sizeof(float)));

    uint32_t* d_seed_ptr = nullptr;
    if (seeds != nullptr && num_seeds_per_query > 0) {
        CUDA_CHECK(cudaMalloc(&d_seed_ptr, num_queries * num_seeds_per_query * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_seed_ptr, seeds, num_queries * num_seeds_per_query * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    uint32_t* d_seeds;
    num_seeds_per_query = this->search_params_.itopk_size;
    CUDA_CHECK(cudaMalloc(&d_seeds, num_queries * num_seeds_per_query * sizeof(uint32_t)));

    {
        // 准备范围查找用的种子，从start_ts到end_ts范围内收集所有ID 
        // 只给第一个准备，剩余的广播
        // 首先收集所有的符合时间戳范围的ID
        std::vector<uint32_t> host_seeds;
        for (const auto& [ts, ids] : ts_to_ids_) {
            if (ts >= start_ts && ts < end_ts) {
                host_seeds.insert(host_seeds.end(), ids.begin(), ids.end());
            }
        }
        // 从收集到的ID中，随机采样num_seeds_per_query个作为种子
        std::vector<uint32_t> sampled_seeds;
        if (host_seeds.size() <= num_seeds_per_query) {
            sampled_seeds = host_seeds;
        } else {
            std::sample(host_seeds.begin(), host_seeds.end(), std::back_inserter(sampled_seeds),
                        num_seeds_per_query, std::mt19937{std::random_device{}()});
        }

        // 先广播，再拷贝
        std::vector<uint32_t> all_seeds(num_queries * num_seeds_per_query, 0);
        for (size_t i = 0; i < num_queries; ++i) {
            std::copy(sampled_seeds.begin(), sampled_seeds.end(), all_seeds.begin() + i * num_seeds_per_query);
        }
        CUDA_CHECK(cudaMemcpy(d_seeds, all_seeds.data(), all_seeds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // printf("[Debug] Querying range ts [%llu, %llu] with %zu seeds per query.\n", start_ts, end_ts, num_seeds_per_query);
    // printf("range is %f\n", range_radius);
    // printf("search params is itopk_size=%u, search_width=%u, max_iterations=%u, hash_bitlen=%u\n",
        //    search_params_.itopk_size,
        //    search_params_.search_width,
        //    search_params_.max_iterations,
        //    search_params_.hash_bitlen);
    if (range_radius < 0.99) {
        cagra::search_bucket_range((float*)d_data_vmm_->data(), 
                        this->dim_,
                        current_size_, 
                        (uint32_t*)d_graph_vmm_->data(), 
                        (uint64_t*)this->d_ts_vmm_->data(),
                        graph_degree_, 
                        graph_degree_,
                        d_queries, 
                        num_queries, 
                        k,
                        start_ts,
                        end_ts, 
                        search_params_, // 使用成员变量
                        d_indices, 
                        d_dists,
                    d_seeds,
                    num_seeds_per_query);
    } else {
        // printf("[Debug] Performing standard search without range filtering.\n");
        cagra::search((float*)d_data_vmm_->data(), 
                    current_size_, 
                    this->dim_,
                    (uint32_t*)d_graph_vmm_->data(), 
                    graph_degree_, 
                    d_queries, 
                    num_queries, 
                    k, 
                    search_params_, // 使用成员变量
                    d_indices, 
                    d_dists,
                    d_seed_ptr,
                    num_seeds_per_query);
    }

    CUDA_CHECK(cudaMemcpy(host_indices, d_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dists, d_dists, num_queries * k * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaFree(d_seeds));
}

// =========================================================
// 5. Save & Load
// =========================================================
void CagraIndex::save(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for save");

    // 因为更新不会同步数据到Host端，save前要把图数据同步到Host
    size_t graph_bytes = current_size_ * graph_degree_ * sizeof(uint32_t);
    CUDA_CHECK(cudaMemcpy(h_graph_.data(), d_graph_vmm_->data(), graph_bytes, cudaMemcpyDeviceToHost));

    out.write((char*)&dim_, sizeof(dim_));
    out.write((char*)&graph_degree_, sizeof(graph_degree_));
    out.write((char*)&current_size_, sizeof(current_size_));

    if (current_size_ > 0) {
        out.write((char*)h_data_.data(), h_data_.size() * sizeof(float));
        out.write((char*)h_graph_.data(), h_graph_.size() * sizeof(uint32_t));
    }
}

void CagraIndex::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for load");

    in.read((char*)&dim_, sizeof(dim_));
    in.read((char*)&graph_degree_, sizeof(graph_degree_));
    in.read((char*)&current_size_, sizeof(current_size_));

    h_data_.resize(current_size_ * dim_);
    h_graph_.resize(current_size_ * graph_degree_);

    if (current_size_ > 0) {
        in.read((char*)h_data_.data(), h_data_.size() * sizeof(float));
        in.read((char*)h_graph_.data(), h_graph_.size() * sizeof(uint32_t));

        // Restore VMM
        size_t data_bytes = current_size_ * dim_ * sizeof(float);
        size_t graph_bytes = current_size_ * graph_degree_ * sizeof(uint32_t);

        d_data_vmm_->resize(data_bytes);
        d_graph_vmm_->resize(graph_bytes);

        CUDA_CHECK(cudaMemcpy(d_data_vmm_->data(), h_data_.data(), data_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_vmm_->data(), h_graph_.data(), graph_bytes, cudaMemcpyHostToDevice));
    }
}

} // namespace cagra