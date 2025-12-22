#include "cagra.cuh"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <algorithm>
// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS
#include <faiss/IndexFlat.h>
// #include <faiss/gpu/StandardGpuResources.h>
// #include <faiss/gpu/GpuIndexFlat.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// 数据加载辅助
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

#include "common.cuh"

int main() {
    CHECK_CUDA(cudaSetDevice(CUDA_DEVICE_ID));

    // ---------------------------------------------------------
    // 1. 配置
    // ---------------------------------------------------------
    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    const size_t BASE_SIZE = 50000;   // 初始图大小
    const size_t INSERT_SIZE = 50000; // 插入数据量
    const size_t TOTAL_SIZE = BASE_SIZE + INSERT_SIZE;
    
    const uint32_t KNN_K = 128;
    const uint32_t GRAPH_K = 32;

    std::cout << "==========================================================" << std::endl;
    std::cout << "CAGRA Incremental Insert Test" << std::endl;
    std::cout << "Base: " << BASE_SIZE << ", Insert: " << INSERT_SIZE << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ---------------------------------------------------------
    // 2. 加载数据
    // ---------------------------------------------------------
    int dim = -1, file_total = -1;
    if (!parseMeta(meta_path, dim, file_total)) return 1;
    
    if (file_total < TOTAL_SIZE) {
        std::cerr << "Error: Not enough data in file." << std::endl;
        return 1;
    }

    int fd = open(bin_path.c_str(), O_RDONLY);
    size_t file_bytes = (size_t)file_total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);

    // 准备 GPU 内存 (预留足够大的空间)
    // 为了模拟真实场景，我们申请 TOTAL_SIZE 的空间，但初始只填 BASE_SIZE
    float* d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, TOTAL_SIZE * dim * sizeof(float)));
    
    // Copy Base Data
    CHECK_CUDA(cudaMemcpy(d_dataset, host_data, BASE_SIZE * dim * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------
    // 3. Build Base Index
    // ---------------------------------------------------------
    std::cout << ">> [Step 1] Building Base Index (" << BASE_SIZE << " nodes)..." << std::endl;
    
    // KNN
    std::vector<uint32_t> h_raw_knn(BASE_SIZE * KNN_K);
    cagra::generate_knn_graph(d_dataset, BASE_SIZE, dim, KNN_K, h_raw_knn.data());
    
    uint32_t* d_raw_knn;
    CHECK_CUDA(cudaMalloc(&d_raw_knn, BASE_SIZE * KNN_K * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_raw_knn, h_raw_knn.data(), BASE_SIZE * KNN_K * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Build
    cagra::BuildParams build_params;
    build_params.intermediate_degree = KNN_K;
    build_params.graph_degree = GRAPH_K;

    uint32_t* d_graph = nullptr;
    cagra::build(d_dataset, BASE_SIZE, d_raw_knn, build_params, &d_graph);
    CHECK_CUDA(cudaFree(d_raw_knn));

    // ---------------------------------------------------------
    // 4. Incremental Insert
    // ---------------------------------------------------------
    std::cout << ">> [Step 2] Inserting " << INSERT_SIZE << " nodes..." << std::endl;

    // 准备 Host Graph 副本 (因为 insert 函数需要在 CPU 上修改图拓扑)
    // 注意：图的大小需要扩容到 TOTAL_SIZE
    std::vector<uint32_t> h_graph(TOTAL_SIZE * GRAPH_K);
    // 拷贝已有的 Base 图
    CHECK_CUDA(cudaMemcpy(h_graph.data(), d_graph, BASE_SIZE * GRAPH_K * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // 显存图扩容 (Realloc)
    // 简单起见，我们释放旧的，申请新的大的
    CHECK_CUDA(cudaFree(d_graph));
    CHECK_CUDA(cudaMalloc(&d_graph, TOTAL_SIZE * GRAPH_K * sizeof(uint32_t)));
    // 把旧数据拷回去 (其实不拷也行，因为 insert 最后会全量覆盖，但为了逻辑完整性)
    CHECK_CUDA(cudaMemcpy(d_graph, h_graph.data(), BASE_SIZE * GRAPH_K * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 准备新数据指针
    const float* h_new_data_ptr = host_data + BASE_SIZE * dim;
    float* d_new_data_ptr = d_dataset + BASE_SIZE * dim;
    
    // Copy New Data to GPU Dataset (追加)
    CHECK_CUDA(cudaMemcpy(d_new_data_ptr, h_new_data_ptr, INSERT_SIZE * dim * sizeof(float), cudaMemcpyHostToDevice));

    cagra::SearchParams search_params;
    search_params.itopk_size = 128;
    search_params.search_width = 4;
    search_params.max_iterations = 50;

    // 调用 Insert
    // 注意：insert 函数内部会使用 d_dataset (大小 num_existing) 进行搜索
    // 并使用 d_new_data_ptr 作为 query
    // 更新 h_graph，最后拷回 d_graph
    // cagra::insert(d_dataset,      // 旧数据起始
    //               BASE_SIZE,      // 旧数据数量
    //               INSERT_SIZE,    // 新数据数量
    //               d_new_data_ptr, // 新数据起始 (Query)
    //               d_graph,        // GPU 图 (全量空间)
    //               h_graph.data(), // CPU 图 (全量空间)
    //               GRAPH_K,
    //               search_params);
    int nn = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < INSERT_SIZE; i+=64) {
        if (i % 2048 == 0) {
            std::cout << "    Inserting node " << i << " / " << INSERT_SIZE << std::endl;
        }
        nn = std::min(64, static_cast<int>(INSERT_SIZE - i));
        cagra::insert(d_dataset,      // 旧数据起始
                      BASE_SIZE + i, // 旧数据数量 (动态增长)
                      nn,             // 新数据数量
                      d_new_data_ptr + i * dim, // 新数据起始 (Query)
                      d_graph,       // GPU 图 (全量空间)
                      h_graph.data(),// CPU 图 (全量空间)
                      GRAPH_K,
                      search_params);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double insert_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << ">> Total Insert Time for " << INSERT_SIZE << " nodes: " << insert_time << " seconds." << std::endl;

    std::cout << ">> Insert Finished. Total Graph Size: " << TOTAL_SIZE << std::endl;

    // ---------------------------------------------------------
    // 5. Verification (Recall Test)
    // ---------------------------------------------------------
    std::cout << ">> [Step 3] Verifying Recall..." << std::endl;

    int num_queries = 100;
    int k = 10;

    // 随机选择 100 个 Query (从插入的数据中选，或者全量选)
    // 这里我们从 全量数据 (3000) 中选，验证整体连通性
    std::vector<float> h_queries(num_queries * dim);
    std::vector<int64_t> gt_indices(num_queries * k);
    std::vector<float> gt_dists(num_queries * k); // unused

    std::mt19937 rng(999);
    std::uniform_int_distribution<int> dist(0, TOTAL_SIZE - 1);

    for (int i = 0; i < num_queries; ++i) {
        int idx = dist(rng);
        std::copy(host_data + idx * dim, host_data + (idx + 1) * dim, h_queries.data() + i * dim);
    }

    float* d_queries;
    CHECK_CUDA(cudaMalloc(&d_queries, num_queries * dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries.data(), num_queries * dim * sizeof(float), cudaMemcpyHostToDevice));

    // A. 生成 Ground Truth (FAISS Flat L2 on CPU)
    {
        std::cout << ">> [GT] Generating Ground Truth using FAISS Flat L2 (CPU)..." << std::endl;
        
        // 使用 CPU 索引，绝对可靠
        faiss::IndexFlatL2 flat_index(dim); 
        
        // 添加数据：直接使用 mmap 映射的 host_data
        // 注意：数据量是 TOTAL_SIZE
        flat_index.add(TOTAL_SIZE, host_data);
        
        // 搜索：直接使用 Host 端 query 向量
        std::vector<float> gt_dists(num_queries * k); // 必须是 float
        std::vector<int64_t> gt_indices_temp(num_queries * k); // FAISS CPU 输出 int64 (idx_t)

        // 搜索 Top-K
        flat_index.search(num_queries, h_queries.data(), k, gt_dists.data(), gt_indices_temp.data());
        
        // 保存结果到 gt_indices (vector<int64_t>)
        gt_indices = gt_indices_temp;

        // 输出部分 GT 结果供调试
        std::cout << "Ground Truth Indices for first 3 queries:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << "Query " << i << ": ";
            for (int j = 0; j < k; ++j) {
                std::cout << gt_indices[i * k + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // B. CAGRA Search
    int64_t* d_out_indices;
    float* d_out_dists;
    CHECK_CUDA(cudaMalloc(&d_out_indices, num_queries * k * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_out_dists, num_queries * k * sizeof(float)));

    // 注意：Search 时使用 TOTAL_SIZE
    cagra::search(d_dataset, TOTAL_SIZE, d_graph, GRAPH_K, 
                  d_queries, num_queries, k, search_params, d_out_indices, d_out_dists);

    std::vector<int64_t> h_out_indices(num_queries * k);
    CHECK_CUDA(cudaMemcpy(h_out_indices.data(), d_out_indices, num_queries * k * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // C. Calculate Recall
    int correct = 0;
    for (int i = 0; i < num_queries; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(gt_indices[i * k + j]);

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(h_out_indices[i * k + j])) correct++;
        }
    }

    double recall = 100.0 * correct / (num_queries * k);
    std::cout << "Recall@" << k << ": " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;

    // Clean up
    munmap((void*)host_data, file_bytes);
    close(fd);
    CHECK_CUDA(cudaFree(d_dataset));
    CHECK_CUDA(cudaFree(d_graph));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_out_indices));
    CHECK_CUDA(cudaFree(d_out_dists));

    if (recall > 90.0) {
        std::cout << "PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
}