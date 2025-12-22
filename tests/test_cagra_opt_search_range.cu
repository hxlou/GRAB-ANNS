/**
 * TODO: 因为设备1被占用，还未实际进行测试
 */

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

// FAISS (CPU 版本，用于动态生成 Range GT)
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

#include "common.cuh"

int main() {
    CHECK_CUDA(cudaSetDevice(CUDA_DEVICE_ID));

    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    std::cout << "==========================================================" << std::endl;
    std::cout << "CagraIndexOpt RANGE Query Test (Time Filtering)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. 加载数据
    int dim = 1024, file_total = 0;
    parseMeta(meta_path, dim, file_total);
    
    int total = 40000; 
    if (file_total < total) total = file_total;

    // 分桶
    size_t num_buckets = 8;
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

    // 3. 构建索引
    std::cout << ">> [Build] Building Index..." << std::endl;
    cagra::CagraIndexOpt index(dim, 32); 
    index.setBuildParams(64, 32); // KNN=64, Graph=32
    index.add(total, host_data, timestamps.data());
    
    Timer timer;
    index.build();
    std::cout << "   Build Time: " << timer.elapsed_ms() << " ms" << std::endl;

    // 4. 随机 Range 测试配置
    const int TEST_ITERS = 100;
    const int QUERIES_PER_ITER = 32;
    const int K = 10;

    // 搜索参数：启用宽搜和多跳，因为 Range Query 需要跨桶
    index.setQueryParams(
        256,  // itopk (加大一点，防止过滤掉太多)
        4,    // search_width
        0,    // min_iter
        50,   // max_iter (跨桶需要更多步数)
        14    // hash_bitlen
    );

    std::mt19937 rng(5678);

    long long total_queries_cnt = 0;
    long long total_hits = 0;
    long long total_bound_errors = 0;
    double total_search_time = 0.0;

    std::cout << "\n>> [Test] Starting Random Range Tests..." << std::endl;
    std::cout << "   Progress: ";

    for (int iter = 0; iter < TEST_ITERS; ++iter) {
        if (iter % 10 == 0) std::cout << "." << std::flush;

        // A. 随机生成范围 [start, end)
        // start: 0 ~ num_buckets-2
        // length: 2 ~ 10 (随机跨度)
        int start_bucket = std::uniform_int_distribution<int>(0, num_buckets - 2)(rng);
        int max_len = std::min((int)num_buckets - start_bucket, 10); // 最大跨度10个桶
        int len = std::uniform_int_distribution<int>(1, max_len)(rng);
        int end_bucket = start_bucket + len;

        // 计算该范围的数据 ID 区间
        size_t range_global_start = start_bucket * bucket_size;
        size_t range_global_end = std::min((size_t)end_bucket * bucket_size, (size_t)total);
        size_t range_len = range_global_end - range_global_start;
        const float* range_data_ptr = host_data + range_global_start * dim;

        // B. 从该范围内随机选 Query
        std::vector<float> queries(QUERIES_PER_ITER * dim);
        std::uniform_int_distribution<int> q_dist(0, range_len - 1);
        
        for (int i = 0; i < QUERIES_PER_ITER; ++i) {
            int local_idx = q_dist(rng);
            std::copy(range_data_ptr + local_idx * dim,
                      range_data_ptr + (local_idx + 1) * dim,
                      queries.data() + i * dim);
        }

        // C. 动态生成 Range GT (CPU FAISS)
        // 只把范围内的数据加入 Index
        std::vector<int64_t> gt_indices_range(QUERIES_PER_ITER * K);
        std::vector<float> gt_dists(QUERIES_PER_ITER * K);
        {
            faiss::IndexFlatL2 cpu_index(dim);
            cpu_index.add(range_len, range_data_ptr);
            cpu_index.search(QUERIES_PER_ITER, queries.data(), K, gt_dists.data(), gt_indices_range.data());
        }

        // D. 执行 CAGRA Range Query
        std::vector<int64_t> out_indices(QUERIES_PER_ITER * K);
        std::vector<float> out_dists(QUERIES_PER_ITER * K);
        
        timer.reset();
        
        // 【核心】调用 query_range
        // 传入 start_bucket 和 end_bucket 作为过滤条件
        // 传入 local_degree=28 (虽然这里没用上，但这通常用于控制是否只搜Local，
        // 对于Range Query，我们其实是用 Total Degree 在搜，因为要跨桶)
        // 你的接口定义里有 local_degree，我们可以传 32 或 28，
        // 实际上底层 search_bucket_range 用的是 active_degree，
        // 如果要做跨桶搜索，这里的 active_degree 应该传入 32 (total)！
        // 
        // 修正：在 cagraIndexOpt.cu 的 query_range 实现中，
        // 你应该把 active_degree 设为 graph_degree_ (32)，以便利用 Remote Edge 跳跃。
        // 但这里为了接口一致，我们可以传 32。
        
        index.query_range(queries.data(), 
                          QUERIES_PER_ITER, 
                          K, 
                          (uint64_t)start_bucket, 
                          (uint64_t)end_bucket, 
                          out_indices.data(), 
                          out_dists.data(), 
                          32); // 这里传32，允许走Remote Edge
        
        total_search_time += timer.elapsed_ms();

        // E. 验证
        for (int i = 0; i < QUERIES_PER_ITER; ++i) {
            std::unordered_set<int64_t> gt_set;
            // GT 转 Global ID
            for (int j = 0; j < K; ++j) {
                gt_set.insert(gt_indices_range[i * K + j] + range_global_start);
            }

            for (int j = 0; j < K; ++j) {
                int64_t result_gid = out_indices[i * K + j];
                
                // 1. 越界检查 (Filtering Check)
                // 结果必须落在 [start_bucket, end_bucket) 的 ID 范围内
                if (result_gid != -1) {
                    if (result_gid < range_global_start || result_gid >= range_global_end) {
                        total_bound_errors++;
                        if (total_bound_errors == 1) {
                            std::cerr << "\n[FATAL] Filter Failed! Range=[" << start_bucket << ", " << end_bucket << ")"
                                      << " Found GID=" << result_gid << std::endl;
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
    std::cout << "   Filter Errors:     " << total_bound_errors << " (Must be 0)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    munmap((void*)host_data, sz);
    close(fd);

    if (total_bound_errors == 0 && avg_recall > 95.0) {
        std::cout << "PASSED: High recall with perfect filtering." << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: Recall low or filtering broken." << std::endl;
        return 1;
    }
}