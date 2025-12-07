#include "cagraIndex.hpp"
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace cagra {

CagraIndex::CagraIndex(uint32_t dim, uint32_t graph_degree, size_t vmm_max_bytes)
    : dim_(dim), graph_degree_(graph_degree), current_size_(0) 
{
    // VMM 初始化
    size_t data_cap = vmm_max_bytes / 2;
    size_t graph_cap = vmm_max_bytes / 2;
    d_data_vmm_ = std::make_unique<DeviceBufferVMM>(data_cap);
    d_graph_vmm_ = std::make_unique<DeviceBufferVMM>(graph_cap);

    // 1. 构建参数初始化
    build_params_.graph_degree = graph_degree;
    build_params_.intermediate_degree = std::max(128u, graph_degree * 2);

    // 2. 查询参数初始化 (用于 query 接口)
    search_params_.itopk_size = 256;
    search_params_.search_width = 6; // 标准查询平衡参数
    search_params_.max_iterations = 50;
    search_params_.hash_bitlen = 12;

    // 3. 插入参数初始化 (用于 insert 内部搜索，需要更激进以保证连通性)
    insert_params_.itopk_size = 128;
    insert_params_.search_width = 4; // 插入时宽搜，找得更准
    insert_params_.max_iterations = 50;
    insert_params_.hash_bitlen = 12;
}

CagraIndex::~CagraIndex() = default;

// =========================================================
// 1. add: 纯 Host 数据积累
// =========================================================
void CagraIndex::add(size_t num_vectors, const float* add_vectors) {
    if (num_vectors == 0) return;
    
    // 直接追加到 std::vector
    h_data_.insert(h_data_.end(), add_vectors, add_vectors + num_vectors * dim_);
    
    // 更新计数 (此时 graph 还没建，不用管 h_graph_)
    current_size_ = h_data_.size() / dim_;
    
    // std::cout << "[CagraIndex::add] Added " << num_vectors << " vectors. Total: " << current_size_ << std::endl;
}

// =========================================================
// 2. build: 全量构建
// =========================================================
void CagraIndex::build() {
    if (current_size_ == 0) return;
    
    std::cout << ">> [CagraIndex::build] Starting build for " << current_size_ << " vectors..." << std::endl;

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

    std::cout << ">> [CagraIndex::build] Finished." << std::endl;
}

// =========================================================
// 3. insert: 增量插入
// =========================================================
void CagraIndex::insert(size_t new_vectors, const float* insert_vectors) {
    if (new_vectors == 0) return;

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
    int batch = 32;
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
                       float* host_dists) 
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

    cagra::search((float*)d_data_vmm_->data(), 
                  current_size_, 
                  (uint32_t*)d_graph_vmm_->data(), 
                  graph_degree_, 
                  d_queries, 
                  num_queries, 
                  k, 
                  search_params_, // 使用成员变量
                  d_indices, 
                  d_dists);

    CUDA_CHECK(cudaMemcpy(host_indices, d_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_dists, d_dists, num_queries * k * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_dists));
}

// =========================================================
// 5. Save & Load
// =========================================================
void CagraIndex::save(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for save");

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