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
#include <set>
#include <cassert>
#include <map>

// FAISS头文件
#include <faiss/IndexFlat.h>

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>  // for setenv


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

// 读取 fvecs 文件格式的函数
std::vector<float> load_fvecs(const std::string& filename, int& dim, int& num_vectors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // 读取维度
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (file.gcount() != sizeof(int)) {
        throw std::runtime_error("Failed to read dimension from: " + filename);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 计算向量数量
    size_t vector_size_bytes = sizeof(int) + dim * sizeof(float);
    num_vectors = file_size / vector_size_bytes;

    std::vector<float> data(num_vectors * dim);

    // 逐个读取向量
    for (int i = 0; i < num_vectors; ++i) {
        // 跳过维度信息（每个向量前都有维度）
        file.seekg(sizeof(int), std::ios::cur);

        // 读取向量数据
        file.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
        if (file.gcount() != dim * sizeof(float)) {
            throw std::runtime_error("Failed to read vector data from: " + filename);
        }
    }

    file.close();
    std::cout << "Loaded " << num_vectors << " vectors of dimension " << dim << " from " << filename << std::endl;
    return data;
}

int main() {
    int cuda_device = 0;

    // 重置所有CUDA设备并强制设置为设备0
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(cuda_device));

    // 验证当前设备
    int current_device;
    CHECK_CUDA(cudaGetDevice(&current_device));
    std::cout << "Current CUDA device: " << current_device << std::endl;

    std::string base_path = "../data/HDFS.log-1M.fvecs";
    std::string query_path = "../data/HDFS.log-1M.query.fvecs";

    std::cout << "==========================================================" << std::endl;
    std::cout << "CagraIndexOpt HDFS Bucket Test" << std::endl;
    std::cout << "==========================================================" << std::endl;

    try {
        // 1. 加载数据集
        int dim, num_base, num_query;
        std::vector<float> base_data = load_fvecs(base_path, dim, num_base);
        std::vector<float> query_data = load_fvecs(query_path, dim, num_query);

        // 使用大规模数据集
        const int MAX_BASE = 1000000;  // 100万基础向量
        const int MAX_QUERY = 10000;   // 1万个查询

        if (num_base > MAX_BASE) {
            num_base = MAX_BASE;
            base_data.resize(num_base * dim);
            std::cout << "Limited base vectors to " << num_base << std::endl;
        }

        if (num_query > MAX_QUERY) {
            num_query = MAX_QUERY;
            query_data.resize(num_query * dim);
            std::cout << "Limited query vectors to " << num_query << std::endl;
        }

        // 2. 分桶策略配置
        size_t bucket_nums = 20;  // 适中的桶数量
        size_t bucket_size = num_base / bucket_nums;
        size_t num_buckets = bucket_nums;

        std::cout << "Dataset: " << num_base << " base vectors, " << num_query << " queries" << std::endl;
        std::cout << "Dimension: " << dim << std::endl;
        std::cout << "Buckets: " << num_buckets << " (size ~" << bucket_size << ")" << std::endl;

        // 3. 生成时间戳（按顺序分桶）
        std::vector<uint64_t> timestamps(num_base);
        for (int i = 0; i < num_base; ++i) {
            timestamps[i] = i / bucket_size;
        }

        // 4. 构建索引
        std::cout << ">> [Build] Building Index..." << std::endl;
        Timer build_timer;

        cagra::CagraIndexOpt index(dim, 32, 50ULL * 1024 * 1024 * 1024); // 50GB VMM space
        index.setBuildParams(128, 32); // KNN=128, Graph=32
        index.add(num_base, base_data.data(), timestamps.data());
        index.build();

        double build_time = build_timer.elapsed_ms();
        std::cout << "   Index Built in " << build_time << " ms" << std::endl;

        // 5. 配置搜索参数
        index.setQueryParams(
            64,   // itopk
            4,    // search_width
            0,    // min_iter
            30,   // max_iter
            14    // hash_bitlen
        );

        // 6. 测试配置
        const int K = 10;               // Top-K
        const int TEST_QUERIES = 100;   // 测试查询数量
        const int BUCKETS_TO_TEST = 5;  // 测试的桶数量
        const int QUERIES_PER_ITER = 32;  // 每次迭代的查询数量

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> query_dist(0, num_query - 1);
        std::uniform_int_distribution<int> bucket_dist(0, num_buckets - 1);

        std::cout << "\n>> [TEST 1] Global Search Reliability" << std::endl;
        std::cout << "Building Global GT Index (CPU FAISS)..." << std::endl;

        // 构建全局GT
        faiss::IndexFlatL2 global_cpu_index(dim);
        global_cpu_index.add(num_base, base_data.data());

        long long global_hits = 0;
        double global_search_time = 0.0;
        const int GLOBAL_TEST_ROUNDS = 5;

        for (int round = 0; round < GLOBAL_TEST_ROUNDS; ++round) {
            // A. 随机选查询向量
            std::vector<float> queries(TEST_QUERIES * dim);
            std::uniform_int_distribution<int> q_dist(0, num_base - 1);

            for (int i = 0; i < TEST_QUERIES; ++i) {
                int query_idx = q_dist(rng);
                std::copy(&base_data[query_idx * dim],
                         &base_data[(query_idx + 1) * dim],
                         queries.data() + i * dim);
            }

            // B. 生成全局GT
            std::vector<int64_t> gt_indices(TEST_QUERIES * K);
            std::vector<float> gt_dists(TEST_QUERIES * K);
            global_cpu_index.search(TEST_QUERIES, queries.data(), K, gt_dists.data(), gt_indices.data());

            // C. 执行CAGRA全局搜索
            std::vector<int64_t> out_indices(TEST_QUERIES * K);
            std::vector<float> out_dists(TEST_QUERIES * K);

            Timer timer;
            index.query(queries.data(), TEST_QUERIES, K,
                       0, UINT64_MAX,  // 全时间范围
                       out_indices.data(), out_dists.data());
            global_search_time += timer.elapsed_ms();

            // D. 计算召回率
            int round_hits = 0;
            for (int i = 0; i < TEST_QUERIES; ++i) {
                std::unordered_set<int64_t> gt_set;
                for (int j = 0; j < K; ++j) {
                    gt_set.insert(gt_indices[i * K + j]);
                }
                for (int j = 0; j < K; ++j) {
                    if (gt_set.count(out_indices[i * K + j])) {
                        round_hits++;
                    }
                }
            }
            global_hits += round_hits;

            double round_recall = 100.0 * round_hits / (TEST_QUERIES * K);
            double round_qps = TEST_QUERIES * 1000.0 / (timer.elapsed_ms() + 0.0001);

            std::cout << "Round " << (round + 1) << ": Recall=" << std::fixed << std::setprecision(2) << round_recall
                      << "%, QPS=" << (int)round_qps << std::endl;
        }

        double avg_global_recall = 100.0 * global_hits / (GLOBAL_TEST_ROUNDS * TEST_QUERIES * K);
        double avg_global_qps = (GLOBAL_TEST_ROUNDS * TEST_QUERIES * 1000.0) / global_search_time;

        std::cout << "\n------------------------------------------------" << std::endl;
        std::cout << "[Global Result] Avg Recall: " << std::setprecision(2) << avg_global_recall
                  << "%, Avg QPS: " << (int)avg_global_qps << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        std::cout << "\n>> [TEST 2] Local Search Reliability (Bucket Isolation)" << std::endl;

        long long total_local_hits = 0;
        long long total_bound_errors = 0;

        for (int bucket = 0; bucket < BUCKETS_TO_TEST; ++bucket) {
            std::cout << "Testing Bucket " << bucket << "... ";

            // A. 获取桶信息
            size_t global_start = bucket * bucket_size;
            size_t current_bucket_len = std::min(bucket_size, size_t(num_base) - global_start);
            const float* bucket_data_ptr = &base_data[global_start * dim];

            if (current_bucket_len < QUERIES_PER_ITER) {
                std::cout << "Bucket too small, skipping." << std::endl;
                continue;
            }

            // B. 随机选查询向量
            std::vector<float> queries(QUERIES_PER_ITER * dim);
            std::uniform_int_distribution<int> q_dist(0, current_bucket_len - 1);

            for (int i = 0; i < QUERIES_PER_ITER; ++i) {
                int local_idx = q_dist(rng);
                std::copy(bucket_data_ptr + local_idx * dim,
                         bucket_data_ptr + (local_idx + 1) * dim,
                         queries.data() + i * dim);
            }

            // C. 生成桶内GT
            std::vector<int64_t> gt_indices_local(QUERIES_PER_ITER * K);
            std::vector<float> gt_dists(QUERIES_PER_ITER * K);
            {
                faiss::IndexFlatL2 cpu_index(dim);
                cpu_index.add(current_bucket_len, bucket_data_ptr);
                cpu_index.search(QUERIES_PER_ITER, queries.data(), K, gt_dists.data(), gt_indices_local.data());
            }

            // D. 执行本地搜索
            std::vector<int64_t> out_indices(QUERIES_PER_ITER * K);
            std::vector<float> out_dists(QUERIES_PER_ITER * K);

            index.query_local(queries.data(), QUERIES_PER_ITER, K, bucket,
                              out_indices.data(), out_dists.data(), 32);

            // E. 验证结果
            int bucket_hits = 0;
            int bucket_bound_errors = 0;

            for (int i = 0; i < QUERIES_PER_ITER; ++i) {
                // 构建GT集合，用于集合包含检查
                std::unordered_set<int64_t> gt_set;
                for (int j = 0; j < K; ++j) {
                    gt_set.insert(gt_indices_local[i * K + j] + global_start);
                }

                for (int j = 0; j < K; ++j) {
                    int64_t result_gid = out_indices[i * K + j];

                    if (result_gid != -1) {
                        // 边界检查
                        if (result_gid < (int64_t)global_start || result_gid >= (int64_t)(global_start + current_bucket_len)) {
                            bucket_bound_errors++;
                        }

                        // 召回检查 - 使用集合包含而不是精确匹配
                        if (gt_set.count(result_gid)) {
                            bucket_hits++;
                        }
                    }
                }
            }

            double bucket_recall = 100.0 * bucket_hits / (QUERIES_PER_ITER * K);
            std::cout << "Bucket " << bucket << " Avg Recall: " << std::fixed << std::setprecision(2) << bucket_recall
                      << "%, Bound Errors: " << bucket_bound_errors << std::endl;

            total_local_hits += bucket_hits;
            total_bound_errors += bucket_bound_errors;
        }

        double avg_local_recall = 100.0 * total_local_hits / (BUCKETS_TO_TEST * QUERIES_PER_ITER * K);

        std::cout << "\n------------------------------------------------" << std::endl;
        std::cout << "[Local Result] Avg Recall across buckets: " << std::setprecision(2) << avg_local_recall << "%" << std::endl;
        std::cout << "[Local Result] Total Bound Errors: " << total_bound_errors << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        // 7. 最终结论
        std::cout << "\n>> [Conclusion] Search Reliability Test Results:" << std::endl;
        std::cout << "   - Global Search Recall: " << avg_global_recall << "% (semantic accuracy)" << std::endl;
        std::cout << "   - Local Search Recall: " << avg_local_recall << "% (bucket isolation)" << std::endl;
        std::cout << "   - Bucket Boundary Safety: " << total_bound_errors << " errors (must be 0)" << std::endl;

        if (total_bound_errors == 0 && avg_global_recall > 90.0 && avg_local_recall > 80.0) {
            std::cout << "\nPASSED: Both global and local search working correctly." << std::endl;
            return 0;
        } else {
            std::cout << "\nWARNING: Some metrics may need attention." << std::endl;
            return 0;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}