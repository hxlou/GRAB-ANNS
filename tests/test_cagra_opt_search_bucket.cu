#include "cagraIndexOpt.cuh"

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

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS (CPU 版本，用于动态生成 Local GT)
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

int main() {
    int cuda_device = 1; 
    CHECK_CUDA(cudaSetDevice(cuda_device));

    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    std::cout << "==========================================================" << std::endl;
    std::cout << "CagraIndexOpt RANDOM LOCAL Test (100 Iters x 32 Queries)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. 加载数据 (限制 50w)
    int dim = 1024, file_total = 0;
    parseMeta(meta_path, dim, file_total);
    
    int total = 880000; 
    if (file_total < total) total = file_total;

    // 分桶策略：每 5w 一个桶
    size_t bucket_nums = 40;
    size_t bucket_size = total / bucket_nums;
    size_t num_buckets = bucket_nums;

    std::cout << "Dataset: " << total << " vectors. Buckets: " << num_buckets << " (size ~" << bucket_size << ")" << std::endl;

    int fd = open(bin_path.c_str(), O_RDONLY);
    size_t sz = (size_t)total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);

    // 2. 生成时间戳
    std::vector<uint64_t> timestamps(total);
    for (int i = 0; i < total; ++i) {
        timestamps[i] = i / bucket_size;
    }

    // 3. 构建索引
    std::cout << ">> [Build] Building Index..." << std::endl;
    cagra::CagraIndexOpt index(dim, 32); 
    index.setBuildParams(128, 32); // KNN=64, Graph=32
    index.add(total, host_data, timestamps.data());
    index.build();
    std::cout << "   Index Built." << std::endl;

    // 随机找几个输出一下他们的邻居节点
    const uint32_t* data_ptr = index.get_graph();
    for (int test_id = 0; test_id < 5; ++test_id) {
        int gid = test_id * 5000; // 跨度取点
        std::cout << "[Debug] Neighbors of GID=" << gid << ": ";
        for (int n = 0; n < 32; ++n) {
            // 这里直接访问内部数据结构，假设邻居存储在紧邻的位置
            uint32_t neighbor_id = data_ptr[gid * 32 + n];
            std::cout << neighbor_id << " ";
            if (n == 15) std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    // 4. 随机测试配置
    const int TEST_ITERS = 100;     // 测试轮数
    const int QUERIES_PER_ITER = 32; // 每轮查询数
    const int K = 10;                // Top-K

    // 搜索参数 (Local Search 需要宽搜以保证桶内覆盖)
    index.setQueryParams(
        128,  // itopk
        4,    // search_width
        0,    // min_iter
        50,   // max_iter
        14    // hash_bitlen
    );

    std::mt19937 rng(999);
    std::uniform_int_distribution<int> bucket_dist(0, num_buckets - 1);

    long long total_queries_cnt = 0;
    long long total_hits = 0;
    long long total_bound_errors = 0;
    double total_search_time = 0.0;

    std::cout << "\n>> [Test] Starting Random Tests..." << std::endl;
    std::cout << "   Progress: ";

    for (int iter = 0; iter < TEST_ITERS; ++iter) {
        if (iter % 10 == 0) std::cout << "." << std::flush;

        // A. 随机选一个桶
        int target_ts = bucket_dist(rng);
        
        // 计算该桶的数据范围
        size_t global_start = target_ts * bucket_size;
        size_t current_bucket_len = std::min(bucket_size, total - global_start);
        const float* bucket_data_ptr = host_data + global_start * dim;

        // B. 随机选 32 个 Query
        std::vector<float> queries(QUERIES_PER_ITER * dim);
        std::uniform_int_distribution<int> q_dist(0, current_bucket_len - 1);
        
        for (int i = 0; i < QUERIES_PER_ITER; ++i) {
            int local_idx = q_dist(rng);
            std::copy(bucket_data_ptr + local_idx * dim,
                      bucket_data_ptr + (local_idx + 1) * dim,
                      queries.data() + i * dim);
        }

        // C. 动态生成 Local GT (CPU FAISS)
        // 这一步很快，5w 数据毫秒级
        std::vector<int64_t> gt_indices_local(QUERIES_PER_ITER * K);
        std::vector<float> gt_dists(QUERIES_PER_ITER * K);
        {
            faiss::IndexFlatL2 cpu_index(dim);
            cpu_index.add(current_bucket_len, bucket_data_ptr);
            cpu_index.search(QUERIES_PER_ITER, queries.data(), K, gt_dists.data(), gt_indices_local.data());
        }

        // D. 执行 CAGRA Local Query
        std::vector<int64_t> out_indices(QUERIES_PER_ITER * K);
        std::vector<float> out_dists(QUERIES_PER_ITER * K);
        
        Timer timer;
        // 指定 local_degree=28
        index.query_local(queries.data(), QUERIES_PER_ITER, K, target_ts, 
                          out_indices.data(), out_dists.data(), 28);
        total_search_time += timer.elapsed_ms();

        // E. 验证
        for (int i = 0; i < QUERIES_PER_ITER; ++i) {
            std::unordered_set<int64_t> gt_set;
            // GT 转 Global ID
            for (int j = 0; j < K; ++j) {
                gt_set.insert(gt_indices_local[i * K + j] + global_start);
            }

            for (int j = 0; j < K; ++j) {
                int64_t result_gid = out_indices[i * K + j];
                
                // 1. 越界检查
                if (result_gid != -1) {
                    if (result_gid < global_start || result_gid >= global_start + current_bucket_len) {
                        total_bound_errors++;
                        // 严重错误，打印一次即可
                        if (1) {
                            std::cerr << "\n[FATAL] Boundary Error! TS=" << target_ts 
                                      << " Found GID=" << result_gid 
                                      << " Range=[" << global_start << ", " << global_start + current_bucket_len << ")" << std::endl;
                        }
                    }
                }

                // 2. 召回统计
                if (gt_set.count(result_gid)) {
                    total_hits++;
                }
            }
        }
        total_queries_cnt += QUERIES_PER_ITER;
    }
    std::cout << " Done." << std::endl;

    // 统计报告
    double avg_recall = 100.0 * total_hits / (total_queries_cnt * K);
    double avg_latency = total_search_time / total_queries_cnt;
    
    std::cout << "\n------------------------------------------------" << std::endl;
    std::cout << "Summary (100 Iters, 3200 Queries)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "   Total Search Time: " << total_search_time << " ms" << std::endl;
    std::cout << "   Avg QPS:           " << std::fixed << std::setprecision(2) << (total_queries_cnt * 1000.0 / total_search_time) << std::endl;
    std::cout << "   Avg Recall@" << K << ":      " << avg_recall << "%" << std::endl;
    std::cout << "   Bound Errors:      " << total_bound_errors << " (Must be 0)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    munmap((void*)host_data, sz);
    close(fd);

    if (total_bound_errors == 0 && avg_recall > 95.0) {
        std::cout << "PASSED: Robust local search." << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: Issues detected." << std::endl;
        return 1;
    }
}