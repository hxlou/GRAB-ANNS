#include "cagra.cuh"
#include "config.cuh" 

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cassert>

// 系统库用于 mmap
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS 用于生成 Ground Truth
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>

// =============================================================================
// 辅助宏与计时器
// =============================================================================
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

class Timer {
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// 数据加载辅助函数
// =============================================================================

// 简单的 JSON 解析，提取 dim 和 total
bool parseMeta(const std::string& path, int& dim, int& total) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    std::string content;
    while (std::getline(file, line)) content += line;

    // 极其简陋的解析，仅寻找 "dim": 1024 和 "total": NNN
    // 假设 json 格式比较标准
    try {
        auto parse_val = [&](const std::string& key) -> int {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return -1;
            size_t start = content.find(":", pos) + 1;
            size_t end = content.find(",", start);
            if (end == std::string::npos) end = content.find("}", start);
            return std::stoi(content.substr(start, end - start));
        };

        dim = parse_val("dim");
        // meta json 里的 key 可能是 "total" 或者 "count"，根据实际情况调整
        total = parse_val("total"); 
        if (total == -1) total = parse_val("count"); // 尝试另一个 key
        
        return (dim > 0 && total > 0);
    } catch (...) {
        return false;
    }
}

// =============================================================================
// 主测试流程
// =============================================================================
int main() {
    // 1. 文件路径配置
    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    std::cout << "==========================================================" << std::endl;
    std::cout << "CAGRA Real Data Test (HotpotQA)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 2. 解析元数据
    int dim = -1, vector_total = -1;
    if (!parseMeta(meta_path, dim, vector_total)) {
        std::cerr << "Error: Failed to parse meta file: " << meta_path << std::endl;
        // 如果解析失败，您可以手动硬编码尝试
        // dim = 1024; vector_total = ...;
        return 1;
    }

    std::cout << "Dataset Info: " << vector_total << " vectors, " << dim << " dims." << std::endl;

    // 检查维度是否匹配我们的编译时常量
    if (dim != cagra::config::DIM) {
        std::cerr << "Error: Dataset dim (" << dim << ") != Compiled DIM (" 
                  << cagra::config::DIM << "). Please update config.cuh." << std::endl;
        return 1;
    }

    // 3. mmap 加载数据
    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error: Failed to open bin file: " << bin_path << std::endl;
        return 1;
    }

    size_t file_size = (size_t)vector_total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) {
        std::cerr << "Error: mmap failed." << std::endl;
        close(fd);
        return 1;
    }

    // 4. 数据拷贝到 GPU
    // 我们使用全部数据构建索引
    float* d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, file_size));
    CHECK_CUDA(cudaMemcpy(d_dataset, host_data, file_size, cudaMemcpyHostToDevice));

    std::cout << ">> Data loaded to GPU." << std::endl;

    // 5. 准备查询 (Query)
    // 随机选择 1 个向量作为查询
    int num_queries = 1;
    int k = 20; // Top-K

    std::vector<float> h_queries(num_queries * dim);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, vector_total - 1);
    std::vector<int> query_indices;

    for (int i = 0; i < num_queries; ++i) {
        int idx = dist(rng);
        query_indices.push_back(idx); // 记录下原本的 ID，方便 Debug
        std::copy(host_data + idx * dim, 
                  host_data + (idx + 1) * dim, 
                  h_queries.data() + i * dim);
    }

    float* d_queries;
    CHECK_CUDA(cudaMalloc(&d_queries, num_queries * dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries.data(), num_queries * dim * sizeof(float), cudaMemcpyHostToDevice));

    // ==========================================
    // 6. 生成 Ground Truth (使用 FAISS Flat)
    // ==========================================
    std::cout << ">> [GT] Generating Ground Truth using FAISS Flat IP..." << std::endl;
    
    // FAISS 结果缓冲区
    int64_t* d_gt_indices;
    float* d_gt_dists;
    CHECK_CUDA(cudaMalloc(&d_gt_indices, num_queries * k * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_gt_dists, num_queries * k * sizeof(float)));

    {
        faiss::gpu::StandardGpuResources res;
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;
        
        // 使用 L2 距离，因为 CAGRA 实现的是 L2
        faiss::gpu::GpuIndexFlatIP flat_index(&res, dim, config);
        
        flat_index.add(vector_total, d_dataset); // d_dataset 已经在 GPU 上
        flat_index.search(num_queries, d_queries, k, d_gt_dists, d_gt_indices);
    }
    
    // 拷回 Host 备用
    std::vector<int64_t> h_gt_indices(num_queries * k);
    CHECK_CUDA(cudaMemcpy(h_gt_indices.data(), d_gt_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_gt_indices));
    CHECK_CUDA(cudaFree(d_gt_dists));

    // ==========================================
    // 7. CAGRA Pipeline
    // ==========================================
    Timer timer;
    
    // 参数
    const uint32_t KNN_K = 128;
    const uint32_t GRAPH_K = 32; // CAGRA Degree

    // Step A: 生成 KNN 图
    std::cout << ">> [CAGRA] Step 1: Generating KNN Graph..." << std::endl;
    std::vector<uint32_t> h_raw_knn(vector_total * KNN_K);
    
    timer.reset();
    cagra::generate_knn_graph(d_dataset, vector_total, dim, KNN_K, h_raw_knn.data());
    std::cout << "   KNN Gen Time: " << timer.elapsed_ms() << " ms" << std::endl;

    // 拷贝到 GPU 供 build 使用
    uint32_t* d_raw_knn;
    CHECK_CUDA(cudaMalloc(&d_raw_knn, vector_total * KNN_K * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_raw_knn, h_raw_knn.data(), vector_total * KNN_K * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Step B: 构建索引
    std::cout << ">> [CAGRA] Step 2: Building Index..." << std::endl;
    cagra::BuildParams build_params;
    build_params.intermediate_degree = KNN_K;
    build_params.graph_degree = GRAPH_K;

    uint32_t* d_cagra_graph = nullptr;
    
    timer.reset();
    cagra::build(d_dataset, vector_total, d_raw_knn, build_params, &d_cagra_graph);
    std::cout << "   Build Time: " << timer.elapsed_ms() << " ms" << std::endl;

    CHECK_CUDA(cudaFree(d_raw_knn));

    // Step C: 搜索
    std::cout << ">> [CAGRA] Step 3: Searching..." << std::endl;
    int64_t* d_out_indices;
    float* d_out_dists;
    CHECK_CUDA(cudaMalloc(&d_out_indices, num_queries * k * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_out_dists, num_queries * k * sizeof(float)));

    cagra::SearchParams search_params;
    search_params.itopk_size = 128;
    search_params.search_width = 20;
    search_params.max_iterations = 10;

    // 预热 (Warm up)
    cagra::search(d_dataset, vector_total, d_cagra_graph, GRAPH_K, 
                  d_queries, num_queries, k, search_params, d_out_indices, d_out_dists);

    // 正式计时
    timer.reset();
    cagra::search(d_dataset, vector_total, d_cagra_graph, GRAPH_K, 
                  d_queries, num_queries, k, search_params, d_out_indices, d_out_dists);
    double search_time = timer.elapsed_ms();

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Search Time: " << search_time << " ms" << std::endl;
    std::cout << "QPS:         " << (num_queries * 1000.0 / search_time) << std::endl;
    std::cout << "Latency:     " << (search_time / num_queries) << " ms/query" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    // ==========================================
    // 8. 计算召回率 (Recall@K)
    // ==========================================
    std::vector<int64_t> h_out_indices(num_queries * k);
    CHECK_CUDA(cudaMemcpy(h_out_indices.data(), d_out_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int correct_cnt = 0;
    for (int i = 0; i < num_queries; ++i) {
        // 构建 GT 集合 (Set)
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) {
            gt_set.insert(h_gt_indices[i * k + j]);
        }

        // 检查 CAGRA 结果有多少在 GT 中
        int hit = 0;
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(h_out_indices[i * k + j])) {
                hit++;
            }
        }
        correct_cnt += hit;
    }

    double recall = (double)correct_cnt / (num_queries * k);
    
    std::cout << "Recall@" << k << ": " << std::fixed << std::setprecision(4) << recall * 100.0 << "%" << std::endl;

    // 清理资源
    munmap((void*)host_data, file_size);
    close(fd);
    CHECK_CUDA(cudaFree(d_dataset));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_cagra_graph));
    CHECK_CUDA(cudaFree(d_out_indices));
    CHECK_CUDA(cudaFree(d_out_dists));

    return 0;
}