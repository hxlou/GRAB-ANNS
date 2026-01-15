#include "cagra.cuh"

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
bool parseMeta(const std::string& path, int& dim, int& total) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line, content;
    while (std::getline(file, line)) content += line;
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
        total = parse_val("total"); 
        if (total == -1) total = parse_val("count");
        return (dim > 0 && total > 0);
    } catch (...) { return false; }
}

// =============================================================================
// 主测试流程
// =============================================================================

#include "common.cuh"

int main() {
    CHECK_CUDA(cudaSetDevice(CUDA_DEVICE_ID));

    // ---------------------------------------------------------
    // 1. 配置
    // ---------------------------------------------------------
    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    // 测试参数
    int num_queries = 32;
    int k = 20; // Top-K

    // CAGRA 参数
    const uint32_t KNN_K = 128;      // 初始图度数 (越大构建越慢，但质量越好)
    const uint32_t GRAPH_K = 64;     // 最终图度数 (通常 32 或 64)

    std::cout << "==========================================================" << std::endl;
    std::cout << "CAGRA Real Data Test (HotpotQA)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ---------------------------------------------------------
    // 2. 加载数据
    // ---------------------------------------------------------
    int dim = -1, vector_total = -1;
    if (!parseMeta(meta_path, dim, vector_total)) {
        std::cerr << "Error: Failed to parse meta file." << std::endl;
        return 1;
    }
    
    // 如果数据量太大，为了测试速度可以截断，例如只测前 5万
    vector_total = std::min(vector_total, 50000); 

    std::cout << "Dataset Info: " << vector_total << " vectors, " << dim << " dims." << std::endl;

    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) { std::cerr << "Error opening bin file." << std::endl; return 1; }
    size_t file_size = (size_t)vector_total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) { std::cerr << "Error mmap." << std::endl; return 1; }

    float* d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, file_size));
    CHECK_CUDA(cudaMemcpy(d_dataset, host_data, file_size, cudaMemcpyHostToDevice));
    std::cout << ">> Data loaded to GPU." << std::endl;

    // ---------------------------------------------------------
    // 3. 准备查询 (Query) - 记录原始 ID 以便查图
    // ---------------------------------------------------------
    std::vector<float> h_queries(num_queries * dim);
    std::vector<int> query_indices; // 【关键】记录 Query 对应数据集中的哪一行
    
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, vector_total - 1);
    std::unordered_set<int> selected;

    for (int i = 0; i < num_queries;) {
        int idx = dist(rng);
        if (selected.find(idx) == selected.end()) {
            selected.insert(idx);
            query_indices.push_back(idx); // 记录 ID
            std::copy(host_data + idx * dim, 
                      host_data + (idx + 1) * dim, 
                      h_queries.data() + i * dim);
            i++;
        }
    }

    float* d_queries;
    CHECK_CUDA(cudaMalloc(&d_queries, num_queries * dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries.data(), num_queries * dim * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------
    // 4. 生成 Ground Truth (FAISS Flat L2)
    // ---------------------------------------------------------
    std::cout << ">> [GT] Generating Ground Truth using FAISS Flat L2..." << std::endl;
    int64_t* d_gt_indices;
    float* d_gt_dists;
    CHECK_CUDA(cudaMalloc(&d_gt_indices, num_queries * k * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_gt_dists, num_queries * k * sizeof(float)));

    {
        faiss::gpu::StandardGpuResources res;
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 1;
        faiss::gpu::GpuIndexFlatL2 flat_index(&res, dim, config); // 必须用 L2
        flat_index.add(vector_total, d_dataset); 
        flat_index.search(num_queries, d_queries, k, d_gt_dists, d_gt_indices);
    }
    
    std::vector<int64_t> h_gt_indices(num_queries * k);
    CHECK_CUDA(cudaMemcpy(h_gt_indices.data(), d_gt_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_gt_indices));
    CHECK_CUDA(cudaFree(d_gt_dists));

    // ---------------------------------------------------------
    // 5. CAGRA Pipeline
    // ---------------------------------------------------------
    Timer timer;

    // Step A: KNN
    std::cout << ">> [CAGRA] Step 1: Generating KNN Graph..." << std::endl;
    std::vector<uint32_t> h_raw_knn(vector_total * KNN_K);
    timer.reset();
    cagra::generate_knn_graph(d_dataset, vector_total, dim, KNN_K, h_raw_knn.data());
    std::cout << "   KNN Time: " << timer.elapsed_ms() << " ms" << std::endl;

    uint32_t* d_raw_knn;
    CHECK_CUDA(cudaMalloc(&d_raw_knn, vector_total * KNN_K * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_raw_knn, h_raw_knn.data(), vector_total * KNN_K * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Step B: Build
    std::cout << ">> [CAGRA] Step 2: Building Index (Prune/Rev/Merge)..." << std::endl;
    cagra::BuildParams build_params;
    build_params.intermediate_degree = KNN_K;
    build_params.graph_degree = GRAPH_K;

    uint32_t* d_cagra_graph = nullptr;
    timer.reset();
    cagra::build(d_dataset, vector_total, d_raw_knn, build_params, &d_cagra_graph);
    std::cout << "   Build Time: " << timer.elapsed_ms() << " ms" << std::endl;
    CHECK_CUDA(cudaFree(d_raw_knn));

    // ---------------------------------------------------------
    // [重要] 6. 图质量诊断 (Graph Quality Check)
    // 检查 GT 是否存在于构建好的图中
    // ---------------------------------------------------------
    std::cout << ">> [Diagnosis] Checking Graph Theoretical Max Recall..." << std::endl;
    std::vector<uint32_t> h_final_graph(vector_total * GRAPH_K);
    CHECK_CUDA(cudaMemcpy(h_final_graph.data(), d_cagra_graph, 
                          vector_total * GRAPH_K * sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost));

    long long graph_hit_count = 0;
    long long total_gt_edges = num_queries * k;

    for (int i = 0; i < num_queries; ++i) {
        // 拿到当前 Query 在图中的 ID
        int query_node_id = query_indices[i];
        
        // 拿到这个节点的邻居列表
        const uint32_t* neighbors = h_final_graph.data() + query_node_id * GRAPH_K;
        
        // 拿到这个 Query 的真值
        for (int j = 0; j < k; ++j) {
            int64_t gt_id = h_gt_indices[i * k + j];
            
            // 检查 gt_id 是否是 neighbors 之一
            // (这里我们检查的是 Query 节点是否直接连接了 GT 节点，
            //  实际上 CAGRA 是多跳搜索，即使不直连也能搜到，
            //  但如果不直连，说明图的质量不够“一步到位”，
            //  这个指标低不代表搜不到，但代表图结构不够完美)
            // 
            // 修正理解：Graph Quality Check 更有意义的是看 GT 节点是否在图中是连通的，
            // 或者反过来：检查 GT 中的节点，有多少比例出现在了“初始 KNN 图”或者“最终图”中。
            // 
            // 但最硬核的检查是：如果这是 Self-Search，那么 GT 就是自身。
            // 我们这里做的是 Random Query (Subsample)，所以 Query 本身就是图里的点。
            // 我们检查的是：这个点周围的邻居，是否包含了它的 GT。
            for (int deg = 0; deg < GRAPH_K; ++deg) {
                if (neighbors[deg] == (uint32_t)gt_id) {
                    graph_hit_count++;
                    break;
                }
            }
        }
    }
    double graph_recall = 100.0 * graph_hit_count / total_gt_edges;
    std::cout << "   Graph Neighbor Overlap with GT: " << std::fixed << std::setprecision(2) 
              << graph_recall << "% (This indicates how 'perfect' the local graph is)" << std::endl;


    // ---------------------------------------------------------
    // 7. Search
    // ---------------------------------------------------------
    std::cout << ">> [CAGRA] Step 3: Searching..." << std::endl;
    int64_t* d_out_indices;
    float* d_out_dists;
    CHECK_CUDA(cudaMalloc(&d_out_indices, num_queries * k * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_out_dists, num_queries * k * sizeof(float)));

    // 【关键】放宽搜索参数以提升召回
    cagra::SearchParams search_params;
    search_params.itopk_size = 128;     // 内部池大小
    search_params.search_width = 8;     // 宽搜：每次扩展 4 个
    search_params.max_iterations = 50;  // 多跳：最多 50 步

    // Warmup
    cagra::search(d_dataset, vector_total, 111, d_cagra_graph, GRAPH_K, 
                  d_queries, num_queries, k, search_params, d_out_indices, d_out_dists);

    timer.reset();
    cagra::search(d_dataset, vector_total, 111, d_cagra_graph, GRAPH_K, 
                  d_queries, num_queries, k, search_params, d_out_indices, d_out_dists);
    double search_time = timer.elapsed_ms();

    // 输出第一个query的结果和对应的gt，调试用
    for (int i = 0; i < k; ++i) {
        int64_t out_idx;
        float out_dist;
        CHECK_CUDA(cudaMemcpy(&out_idx, d_out_indices + i, sizeof(int64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&out_dist, d_out_dists + i, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Query 0 Result " << i << ": ID=" << out_idx << ", Dist=" << out_dist 
                  << ", GT_ID=" << h_gt_indices[i] << std::endl;
    }

    // ---------------------------------------------------------
    // 8. 统计 Search Recall
    // ---------------------------------------------------------
    std::vector<int64_t> h_out_indices(num_queries * k);
    CHECK_CUDA(cudaMemcpy(h_out_indices.data(), d_out_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int correct_cnt = 0;
    for (int i = 0; i < num_queries; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(h_gt_indices[i * k + j]);

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(h_out_indices[i * k + j])) {
                correct_cnt++;
            }
        }
    }
    double search_recall = 100.0 * correct_cnt / (num_queries * k);

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Search Performance Report:" << std::endl;
    std::cout << "  Time:    " << search_time << " ms" << std::endl;
    std::cout << "  QPS:     " << (num_queries * 1000.0 / search_time) << std::endl;
    std::cout << "  Latency: " << (search_time / num_queries) << " ms/query" << std::endl;
    std::cout << "  Recall@" << k << ": " << search_recall << "%" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    // 清理
    munmap((void*)host_data, file_size);
    close(fd);
    CHECK_CUDA(cudaFree(d_dataset));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_cagra_graph));
    CHECK_CUDA(cudaFree(d_out_indices));
    CHECK_CUDA(cudaFree(d_out_dists));

    return 0;
}