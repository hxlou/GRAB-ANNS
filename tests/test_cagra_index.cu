#include "../src/cagraIndex.hpp"

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
#include <memory>
#include <cstring>

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS (仅用于生成真值 GT)
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/IndexFlat.h> // 引入 CPU FAISS 头文件


// -----------------------------------------------------------------------------
// 辅助工具
// -----------------------------------------------------------------------------
#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

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

// 解析 Meta
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

// 计算 Recall
double calculate_recall(size_t num_queries, int k, const int64_t* gt_indices, const int64_t* ann_indices) {
    int correct = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(gt_indices[i * k + j]);
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(ann_indices[i * k + j])) correct++;
        }
    }
    return 100.0 * correct / (num_queries * k);
}

// 生成 GT (使用 CPU FAISS，确保绝对正确)
void generate_ground_truth(const float* dataset, size_t n, int dim, const float* queries, size_t nq, int k, int64_t* gt_indices) {
    // 2. 使用 CPU FAISS
    // 这里的 metric 要和 CAGRA 保持一致 (L2)
    faiss::IndexFlatL2 cpu_index(dim);
    cpu_index.add(n, dataset);
    
    // 3. 搜索
    std::vector<float> cpu_dists(nq * k);
    std::vector<int64_t> cpu_indices(nq * k);
    
    cpu_index.search(nq, queries, k, cpu_dists.data(), cpu_indices.data());

    // 4. 结果拷回输出指针
    // 注意：测试代码里的 gt_indices 是 Host 指针还是 Device 指针？
    // 看你的调用代码，它是 std::vector<int64_t> 的 data()，是 Host 指针。
    // 所以直接 memcpy 即可。
    std::memcpy(gt_indices, cpu_indices.data(), nq * k * sizeof(int64_t));
}

// =============================================================================
// TEST 1: Add -> Build -> Query -> Save
// =============================================================================
void run_test_build(const float* full_data, int dim, size_t n_build, const std::string& index_path) {
    std::cout << "\n[TEST 1] Add (" << n_build << ") -> Build -> Query -> Save" << std::endl;
    const size_t N_QUERY = 32;
    const int K = 10;
    Timer timer;

    // 1. 初始化
    cagra::CagraIndex index(dim); 

    // 2. Add (积累数据)
    std::cout << ">> Adding data..." << std::endl;
    index.add(n_build, full_data);

    // 3. Build (全量构建)
    std::cout << ">> Building index..." << std::endl;
    timer.reset();
    index.build(); // <--- 极简接口：不需要参数
    std::cout << "   Build Time: " << timer.elapsed_ms() << " ms" << std::endl;

    // 4. Query 准备
    std::vector<float> queries(N_QUERY * dim);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n_build - 1);
    for (size_t i = 0; i < N_QUERY; ++i) {
        int idx = dist(rng);
        std::copy(full_data + idx * dim, full_data + (idx + 1) * dim, queries.data() + i * dim);
    }

    // 5. GT
    std::vector<int64_t> gt_indices(N_QUERY * K);
    generate_ground_truth(full_data, n_build, dim, queries.data(), N_QUERY, K, gt_indices.data());

    // 6. Query 执行
    std::cout << ">> Querying..." << std::endl;
    std::vector<int64_t> out_indices(N_QUERY * K);
    std::vector<float> out_dists(N_QUERY * K);

    // timer.reset();
    auto t1 = std::chrono::high_resolution_clock::now();
    // <--- 极简接口：不需要 SearchParams
    index.query(queries.data(), N_QUERY, K, out_indices.data(), out_dists.data());
    // double t_search = timer.elapsed_ms();
    auto t2 =  std::chrono::high_resolution_clock::now();
    double t_search = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 7. 统计
    double recall = calculate_recall(N_QUERY, K, gt_indices.data(), out_indices.data());
    std::cout << "   Time:   " << t_search << " ms" << std::endl;
    std::cout << "   QPS:    " << (N_QUERY * 1000.0 / t_search) << std::endl;
    std::cout << "   Recall: " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;

    // 8. Save
    index.save(index_path);
}

// =============================================================================
// TEST 2: Load -> Insert -> Query (Enhanced)
// =============================================================================
void run_test_insert(const float* full_data, int dim, size_t n_old, size_t n_new, const std::string& index_path) {
    std::cout << "\n[TEST 2] Load -> Insert (" << n_new << ") -> Query" << std::endl;
    const size_t N_TOTAL = n_old + n_new;
    
    // --- 配置测试参数 ---
    const int K = 10;
    const int BATCH_SIZE = 32;      // 单次查询的 Batch 大小
    const int NUM_BATCHES = 100;     // 重复多少轮
    const size_t TOTAL_QUERIES = BATCH_SIZE * NUM_BATCHES; // 总查询数量

    Timer timer;

    // 1. Load
    std::cout << ">> Loading index..." << std::endl;
    cagra::CagraIndex index(dim);
    index.load(index_path);

    if (index.size() != n_old) {
        std::cerr << "Size mismatch! Expected " << n_old << " got " << index.size() << std::endl;
        exit(1);
    }

    // 2. Insert (增量插入)
    std::cout << ">> Inserting new data..." << std::endl;
    const float* new_data = full_data + n_old * dim;
    
    timer.reset();
    index.insert(n_new, new_data);
    double t_insert = timer.elapsed_ms();
    
    std::cout << "   Insert Time: " << t_insert << " ms (" << (t_insert/n_new) << " ms/vec)" << std::endl;

    // -----------------------------------------------------------
    // 3. Query 准备 (生成 TOTAL_QUERIES 个不重复的随机查询)
    // -----------------------------------------------------------
    if (TOTAL_QUERIES > N_TOTAL) {
        std::cerr << "Error: Not enough data for unique queries!" << std::endl;
        return;
    }

    std::cout << ">> Preparing " << TOTAL_QUERIES << " unique queries..." << std::endl;
    std::vector<float> all_queries(TOTAL_QUERIES * dim);
    
    // 生成所有可能的索引并打乱，确保无重复选择
    std::vector<size_t> all_indices(N_TOTAL);
    std::iota(all_indices.begin(), all_indices.end(), 0); // 0, 1, 2, ..., N_TOTAL-1
    
    std::mt19937 rng(1145); // 固定种子保证可复现
    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    // 拷贝打乱后的前 TOTAL_QUERIES 个向量作为查询集
    for (size_t i = 0; i < TOTAL_QUERIES; ++i) {
        size_t data_idx = all_indices[i];
        std::copy(full_data + data_idx * dim, 
                  full_data + (data_idx + 1) * dim, 
                  all_queries.data() + i * dim);
    }

    // 4. GT 生成 (一次性为所有 Query 生成真值，不计入搜索耗时)
    std::cout << ">> Generating Ground Truth..." << std::endl;
    std::vector<int64_t> gt_indices(TOTAL_QUERIES * K);
    generate_ground_truth(full_data, N_TOTAL, dim, all_queries.data(), TOTAL_QUERIES, K, gt_indices.data());

    // -----------------------------------------------------------
    // 5. Query 执行 (循环 N 次，统计总耗时)
    // -----------------------------------------------------------
    std::cout << ">> Querying (" << NUM_BATCHES << " batches x " << BATCH_SIZE << " queries)..." << std::endl;
    
    // 准备接收所有结果的容器
    std::vector<int64_t> all_out_indices(TOTAL_QUERIES * K);
    std::vector<float> all_out_dists(TOTAL_QUERIES * K);

    double total_search_time_ms = 0.0;

    for (int b = 0; b < NUM_BATCHES; ++b) {
        // 定位当前 Batch 的指针
        float* batch_queries = all_queries.data() + b * BATCH_SIZE * dim;
        int64_t* batch_out_indices = all_out_indices.data() + b * BATCH_SIZE * K;
        float* batch_out_dists = all_out_dists.data() + b * BATCH_SIZE * K;

        // 计时并执行查询
        timer.reset();
        index.query(batch_queries, BATCH_SIZE, K, batch_out_indices, batch_out_dists);
        total_search_time_ms += timer.elapsed_ms();
    }

    // 6. 统计结果
    // 计算总召回率 (覆盖所有批次)
    double recall = calculate_recall(TOTAL_QUERIES, K, gt_indices.data(), all_out_indices.data());
    double avg_qps = (TOTAL_QUERIES * 1000.0) / total_search_time_ms;
    double avg_ips = (n_new * 1000.0) / t_insert;
    double avg_latency = total_search_time_ms / NUM_BATCHES; // 平均每个Batch的耗时

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "   Total Queries: " << TOTAL_QUERIES << std::endl;
    std::cout << "   Total Time:    " << total_search_time_ms << " ms" << std::endl;
    std::cout << "   Avg Latency:   " << avg_latency << " ms / batch(" << BATCH_SIZE << ")" << std::endl;
    std::cout << "   QPS:           " << std::fixed << std::setprecision(2) << avg_qps << std::endl;
    std::cout << "   IPS:           " << std::fixed << std::setprecision(2) << avg_ips << std::endl;
    std::cout << "   Avg Recall:    " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

// =============================================================================
// TEST 3: Baseline (Full Build) - Control Group
// =============================================================================
void run_test_full_baseline(const float* full_data, int dim, size_t n_total) {
    std::cout << "\n[TEST 3] Baseline: Full Build (" << n_total << ") -> Query" << std::endl;
    
    // --- 配置测试参数 (与 TEST 2 保持一致) ---
    const int K = 10;
    const int BATCH_SIZE = 32;
    const int NUM_BATCHES = 100;
    const size_t TOTAL_QUERIES = BATCH_SIZE * NUM_BATCHES;

    Timer timer;

    // 1. 初始化 & Add All
    cagra::CagraIndex index(dim); 
    std::cout << ">> Adding " << n_total << " vectors..." << std::endl;
    index.add(n_total, full_data);

    // 2. Build (一次性全量优化)
    std::cout << ">> Building full index..." << std::endl;
    timer.reset();
    index.build(); 
    double t_build = timer.elapsed_ms();
    std::cout << "   Full Build Time: " << t_build << " ms" << std::endl;

    // -----------------------------------------------------------
    // 3. Query 准备 (逻辑完全同 TEST 2，确保公平)
    // -----------------------------------------------------------
    if (TOTAL_QUERIES > n_total) { std::cerr << "Not enough data" << std::endl; return; }

    std::cout << ">> Preparing " << TOTAL_QUERIES << " unique queries..." << std::endl;
    std::vector<float> all_queries(TOTAL_QUERIES * dim);
    std::vector<size_t> all_indices(n_total);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    // 使用相同的种子 1145，确保抽样的 Query 向量和 TEST 2 完全一致！
    // 这样对比 Recall 才有意义。
    std::mt19937 rng(1145); 
    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    for (size_t i = 0; i < TOTAL_QUERIES; ++i) {
        size_t data_idx = all_indices[i];
        std::copy(full_data + data_idx * dim, 
                  full_data + (data_idx + 1) * dim, 
                  all_queries.data() + i * dim);
    }

    // 4. GT (同 TEST 2)
    std::cout << ">> Generating Ground Truth..." << std::endl;
    std::vector<int64_t> gt_indices(TOTAL_QUERIES * K);
    generate_ground_truth(full_data, n_total, dim, all_queries.data(), TOTAL_QUERIES, K, gt_indices.data());

    // -----------------------------------------------------------
    // 5. Query 执行
    // -----------------------------------------------------------
    std::cout << ">> Querying..." << std::endl;
    std::vector<int64_t> all_out_indices(TOTAL_QUERIES * K);
    std::vector<float> all_out_dists(TOTAL_QUERIES * K);

    double total_search_time_ms = 0.0;

    for (int b = 0; b < NUM_BATCHES; ++b) {
        float* batch_queries = all_queries.data() + b * BATCH_SIZE * dim;
        int64_t* batch_out_indices = all_out_indices.data() + b * BATCH_SIZE * K;
        float* batch_out_dists = all_out_dists.data() + b * BATCH_SIZE * K;

        timer.reset();
        index.query(batch_queries, BATCH_SIZE, K, batch_out_indices, batch_out_dists);
        total_search_time_ms += timer.elapsed_ms();
    }

    // 6. 统计
    double recall = calculate_recall(TOTAL_QUERIES, K, gt_indices.data(), all_out_indices.data());
    double avg_qps = (TOTAL_QUERIES * 1000.0) / total_search_time_ms;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "   [Baseline Result]" << std::endl;
    std::cout << "   Total Time:    " << total_search_time_ms << " ms" << std::endl;
    std::cout << "   QPS:           " << std::fixed << std::setprecision(2) << avg_qps << std::endl;
    std::cout << "   Avg Recall:    " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

int main() {
    int cuda_device = 1;
    CHECK_CUDA(cudaSetDevice(cuda_device));


    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";
    std::string index_path = "cagra_workflow.idx";

    int dim = -1, total = -1;
    if (!parseMeta(meta_path, dim, total)) { std::cerr << "Meta parse failed" << std::endl; return 1; }

    // mmap
    int fd = open(bin_path.c_str(), O_RDONLY);
    size_t sz = (size_t)total * dim * sizeof(float);
    const float* data = (const float*)mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);

    // Run Tests
    size_t n_build = 50000;
    size_t n_insert = 750000;
    size_t n_total = n_build + n_insert;


    run_test_build(data, dim, n_build, index_path);
    run_test_insert(data, dim, n_build, n_insert, index_path);

    run_test_full_baseline(data, dim, n_total);

    // Cleanup
    remove(index_path.c_str());
    munmap((void*)data, sz);
    close(fd);
    return 0;
}