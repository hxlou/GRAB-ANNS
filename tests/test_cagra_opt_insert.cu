#include "../src/cagraIndexOpt.cuh"

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
#include <map>
#include <numeric>

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS (CPU 版本)
#include <faiss/IndexFlat.h>

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
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("\"dim\"") != std::string::npos) {
            size_t pos = line.find(":");
            dim = std::stoi(line.substr(pos + 1));
        }
        if (line.find("\"total\"") != std::string::npos || line.find("\"count\"") != std::string::npos) {
            size_t pos = line.find(":");
            total = std::stoi(line.substr(pos + 1));
        }
    }
    if (dim <= 0) dim = 1024; 
    return true;
}

// 通用召回率计算
double calc_recall(size_t nq, int k, const int64_t* gt, const int64_t* res) {
    int correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(gt[i * k + j]);
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(res[i * k + j])) correct++;
        }
    }
    return 100.0 * correct / (nq * k);
}

// 统计结构体
struct TestStats {
    double avg_recall = 0.0;
    double avg_qps = 0.0;
    double avg_latency = 0.0;
    int bound_errors = 0;
};

// =============================================================================
// 主流程
// =============================================================================

#include "common.cuh"

int main() {
    std::cout << "Using CUDA Device ID: " << CUDA_DEVICE_ID << std::endl;
    CHECK_CUDA(cudaSetDevice(CUDA_DEVICE_ID));

    

    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    std::cout << "==========================================================" << std::endl;
    std::cout << "CagraIndexOpt Comprehensive Test (Multi-Round)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    uint32_t init_degree = 128;
    // CAGRA 全局搜索参数 (可以激进一点)
    uint32_t itopk_size = 512;
    uint32_t search_width = 6;
    uint32_t min_iterations = 0;
    uint32_t max_iterations = 100;
    uint32_t hash_bitlen = 16;


    // 1. 加载数据
    int dim = 1024, file_total = 0;
    parseMeta(meta_path, dim, file_total);
    
    // 使用 880000 数据
    int total = 880000; 
    if (file_total < total) total = file_total;

    // 分桶策略：40 个桶 (每个桶约 2.2w)
    size_t num_buckets = 500;
    size_t bucket_size = total / num_buckets;

    std::cout << "Dataset: " << total << " vectors. Buckets: " << num_buckets << " (size ~" << bucket_size << ")" << std::endl;

    int fd = open(bin_path.c_str(), O_RDONLY);
    size_t sz = (size_t)total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);

    // 2. 生成时间戳
    std::vector<uint64_t> timestamps(total);
    for (int i = 0; i < total; ++i) {
        timestamps[i] = i / bucket_size;
    }

    // 3. 数据切分 (50% Build, 50% Insert)
    // 模拟真实的增量场景：每个桶都先有一半数据，后来又插入了一半数据
    std::vector<float> build_data;
    std::vector<uint64_t> build_ts;
    std::vector<float> insert_data;
    std::vector<uint64_t> insert_ts;
    
    build_data.reserve(total/2 * dim);
    insert_data.reserve(total/2 * dim);

    for (int i = 0; i < total; ++i) {
        size_t offset_in_bucket = i % bucket_size;
        const float* vec = host_data + i * dim;
        
        if (offset_in_bucket < bucket_size / 2) {
            build_data.insert(build_data.end(), vec, vec + dim);
            build_ts.push_back(timestamps[i]);
        } else {
            insert_data.insert(insert_data.end(), vec, vec + dim);
            insert_ts.push_back(timestamps[i]);
        }
    }

    // 4. 初始化与构建
    std::cout << ">> [Setup] Initializing Index..." << std::endl;
    cagra::CagraIndexOpt index(dim, init_degree);
    index.setBuildParams(init_degree * 2, init_degree);

    // Phase 1: Build
    Timer timer;
    index.add(build_ts.size(), build_data.data(), build_ts.data());
    index.build();
    std::cout << "   Build Time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Phase 2: Insert
    std::cout << ">> [Setup] Inserting Incremental Data..." << std::endl;
    // Insert 参数：宽搜 + 深搜，保证连接质量
    index.setQueryParams(512, 6, 0, 100, 16); 
    timer.reset();
    index.insert(insert_ts.size(), insert_data.data(), insert_ts.data());
    std::cout << "   Insert Time: " << timer.elapsed_ms() << " ms" << std::endl;

    // =========================================================================
    // TEST BLOCK 1: Global Search (Multi-Round)
    // =========================================================================
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "TEST 1: Global Search Reliability" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 准备全局 GT 索引 (只建一次)
    std::cout << ">> Building Global GT Index (CPU FAISS)..." << std::endl;
    faiss::IndexFlatL2 global_gt_index(dim);
    global_gt_index.add(total, index.get_data()); // 全量数据

    const int NUM_ROUNDS_GLOBAL = 5;
    const int QUERIES_PER_ROUND = 512;
    const int K = 20;
    

    index.setQueryParams(itopk_size, search_width, min_iterations, max_iterations, hash_bitlen);
    std::cout << "Global search with params of itopk=" << itopk_size 
              << ", search_width=" << search_width 
              << ", max_iterations=" << max_iterations 
              << ", hash_bitlen=" << hash_bitlen << std::endl;


    TestStats global_stats;

    for (int round = 0; round < NUM_ROUNDS_GLOBAL; ++round) {
        // A. 随机采样 Query
        std::vector<float> queries(QUERIES_PER_ROUND * dim);
        std::mt19937 rng(42 + round); // 每轮不同的随机种子
        std::uniform_int_distribution<int> dist(0, total - 1);
        
        for (int i = 0; i < QUERIES_PER_ROUND; ++i) {
            int idx = dist(rng);
            std::copy(host_data + idx * dim, host_data + (idx + 1) * dim, queries.data() + i * dim);
        }

        // B. 获取 GT
        std::vector<int64_t> gt_indices(QUERIES_PER_ROUND * K);
        std::vector<float> gt_dists(QUERIES_PER_ROUND * K);
        global_gt_index.search(QUERIES_PER_ROUND, queries.data(), K, gt_dists.data(), gt_indices.data());

        // C. CAGRA 搜索
        std::vector<int64_t> out_indices(QUERIES_PER_ROUND * K);
        std::vector<float> out_dists(QUERIES_PER_ROUND * K);

        timer.reset();
        index.query(queries.data(), QUERIES_PER_ROUND, K, 0, UINT64_MAX, out_indices.data(), out_dists.data());
        double time_ms = timer.elapsed_ms();

        // D. 统计
        double recall = calc_recall(QUERIES_PER_ROUND, K, gt_indices.data(), out_indices.data());
        double qps = QUERIES_PER_ROUND * 1000.0 / time_ms;
        
        global_stats.avg_recall += recall;
        global_stats.avg_qps += qps;
        global_stats.avg_latency += time_ms / QUERIES_PER_ROUND;

        std::cout << "   Round " << round + 1 << ": Recall=" << std::fixed << std::setprecision(2) << recall 
                  << "%, QPS=" << (int)qps << std::endl;
    }

    // 平均
    global_stats.avg_recall /= NUM_ROUNDS_GLOBAL;
    global_stats.avg_qps /= NUM_ROUNDS_GLOBAL;
    global_stats.avg_latency /= NUM_ROUNDS_GLOBAL;

    std::cout << ">> [Global Result] Avg Recall: " << global_stats.avg_recall 
              << "%, Avg QPS: " << global_stats.avg_qps << std::endl;


    // =========================================================================
    // TEST BLOCK 2: Local Search (Multi-Round per Bucket)
    // =========================================================================
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "TEST 2: Local Search Reliability (Bucket Isolation)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 选取几个测试桶
    std::vector<int> test_buckets = {0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35};
    const int NUM_ROUNDS_LOCAL = 10;
    
    uint32_t true_test_buckets = 0;
    // CAGRA 局部搜索参数 (需要宽搜以覆盖桶内)
    index.setQueryParams(itopk_size, search_width, min_iterations, max_iterations, hash_bitlen);
    std::cout << "Local search with params of itopk=" << itopk_size 
              << ", search_width=" << search_width 
              << ", max_iterations=" << max_iterations 
              << ", hash_bitlen=" << hash_bitlen << std::endl;

    TestStats total_local_stats;

    for (int target_ts : test_buckets) {
        std::cout << ">> Testing Bucket " << target_ts << "..." << std::endl;
        
        // 1. 准备该桶的全量数据 (Gather)
        std::vector<uint32_t> bucket_gids = index.get_ids_by_timestamp(target_ts);
        size_t current_bucket_count = bucket_gids.size();
        
        if (current_bucket_count == 0) continue;
        true_test_buckets++;
        // Gather 向量
        std::vector<float> bucket_vectors(current_bucket_count * dim);
        const float* index_data_ptr = index.get_data();
        for (size_t i = 0; i < current_bucket_count; ++i) {
            uint32_t gid = bucket_gids[i];
            const float* src = index_data_ptr + (size_t)gid * dim;
            std::copy(src, src + dim, bucket_vectors.data() + i * dim);
        }

        // 2. 构建 Local GT Index
        faiss::IndexFlatL2 local_gt_index(dim);
        local_gt_index.add(current_bucket_count, bucket_vectors.data());

        // 3. 多轮测试
        TestStats bucket_stats;
        
        // 建立当前桶的 Global ID 集合 (用于快速判断越界)
        std::unordered_set<uint32_t> valid_gids_set(bucket_gids.begin(), bucket_gids.end());

        for (int round = 0; round < NUM_ROUNDS_LOCAL; ++round) {
            // A. 随机 Query (从桶内数据采样)
            std::vector<float> queries(QUERIES_PER_ROUND * dim);
            std::mt19937 rng(target_ts * 1000 + round);
            std::uniform_int_distribution<int> dist(0, current_bucket_count - 1);

            for (int i = 0; i < QUERIES_PER_ROUND; ++i) {
                int local_idx = dist(rng);
                std::copy(bucket_vectors.data() + local_idx * dim,
                          bucket_vectors.data() + (local_idx + 1) * dim,
                          queries.data() + i * dim);
            }

            // B. 获取 GT
            std::vector<int64_t> gt_local_indices(QUERIES_PER_ROUND * K);
            std::vector<float> gt_dists(QUERIES_PER_ROUND * K);
            local_gt_index.search(QUERIES_PER_ROUND, queries.data(), K, gt_dists.data(), gt_local_indices.data());

            // 转换 GT: Local Index -> Global Index
            std::vector<int64_t> gt_global_indices(QUERIES_PER_ROUND * K);
            for (size_t i = 0; i < gt_global_indices.size(); ++i) {
                gt_global_indices[i] = bucket_gids[gt_local_indices[i]];
            }

            // C. CAGRA 局部搜索
            std::vector<int64_t> out_indices(QUERIES_PER_ROUND * K);
            std::vector<float> out_dists(QUERIES_PER_ROUND * K);

            timer.reset();
            index.query_local(queries.data(), QUERIES_PER_ROUND, K, target_ts, 
                              out_indices.data(), out_dists.data(), 28);
            double time_ms = timer.elapsed_ms();

            // D. 统计
            // 越界检查
            for (auto gid : out_indices) {
                if (gid != -1 && valid_gids_set.find((uint32_t)gid) == valid_gids_set.end()) {
                    bucket_stats.bound_errors++;
                }
            }

            double recall = calc_recall(QUERIES_PER_ROUND, K, gt_global_indices.data(), out_indices.data());
            bucket_stats.avg_recall += recall;
            bucket_stats.avg_qps += (QUERIES_PER_ROUND * 1000.0 / time_ms);
            
        }

        bucket_stats.avg_recall /= NUM_ROUNDS_LOCAL;
        bucket_stats.avg_qps /= NUM_ROUNDS_LOCAL;
        
        std::cout << "   Bucket " << target_ts << " Avg Recall: " << std::fixed << std::setprecision(2) 
                  << bucket_stats.avg_recall << "%, Bound Errors: " << bucket_stats.bound_errors << std::endl;

        total_local_stats.avg_recall += bucket_stats.avg_recall;
        total_local_stats.bound_errors += bucket_stats.bound_errors;
    }
    
    total_local_stats.avg_recall /= true_test_buckets;

    std::cout << ">> [Local Result] Avg Recall across buckets: " << total_local_stats.avg_recall << "%" << std::endl;
    std::cout << ">> [Local Result] Total Bound Errors: " << total_local_stats.bound_errors << std::endl;

    munmap((void*)host_data, sz);
    close(fd);

    bool pass_global = global_stats.avg_recall > 90.0;
    bool pass_local = total_local_stats.avg_recall > 95.0 && total_local_stats.bound_errors == 0;

    if (pass_global && pass_local) {
        std::cout << "\nPASSED: All comprehensive tests passed." << std::endl;
        return 0;
    } else {
        std::cout << "\nFAILED: Performance/Accuracy issues detected." << std::endl;
        return 1;
    }
}