#include "timeStampIndex.cuh"

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
#include <set>

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/IndexFlat.h>

// =============================================================================
// 辅助工具
// =============================================================================
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

// =============================================================================
// 主测试逻辑
// =============================================================================
int main() {
    // 1. 环境配置 (选择空闲显卡)
    int cuda_device = 1; 
    CHECK_CUDA(cudaSetDevice(cuda_device));

    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    std::cout << "==========================================================" << std::endl;
    std::cout << "TimeStampIndex Integration Test (Data Limited to 500k)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 2. 加载数据
    int dim = -1, file_total = -1;
    if (!parseMeta(meta_path, dim, file_total)) { std::cerr << "Meta parse failed" << std::endl; return 1; }
    
    // 【限制数据量】防止 3090Ti 显存不足
    int total = 880000;
    if (file_total < total) total = file_total; 

    std::cout << "Dataset: Using " << total << " vectors, " << dim << " dims." << std::endl;

    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) { std::cerr << "Failed to open bin file" << std::endl; return 1; }
    
    size_t sz = (size_t)total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) { std::cerr << "mmap failed" << std::endl; close(fd); return 1; }

    // 3. 生成时间戳 (模拟)
    // 策略：每 5w 条数据一个时间戳 (0, 1, 2... ~9)
    size_t bucket_count = 50;
    size_t bucket_size = total / bucket_count;
    std::cout << ">> Generating Timestamps (1 timestamp per " << bucket_size << " vectors)..." << std::endl;
    
    std::vector<uint64_t> timestamps(total);
    std::map<uint64_t, int> bucket_counts;
    for (int i = 0; i < total; ++i) {
        timestamps[i] = i / bucket_size;
        bucket_counts[timestamps[i]]++;
    }
    std::cout << "   Total Buckets: " << bucket_counts.size() << std::endl;

    // 4. 准备查询与真值 (Ground Truth)
    const int num_queries = 1000;
    const int k = 20;
    std::cout << ">> Preparing " << num_queries << " queries & Ground Truth..." << std::endl;

    std::vector<float> queries(num_queries * dim);
    std::vector<int64_t> gt_indices(num_queries * k);
    std::vector<float> gt_dists(num_queries * k);

    // 随机采样 Query
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, total - 1);
    
    for (int i = 0; i < num_queries; ++i) {
        int idx = dist(rng);
        std::copy(host_data + idx * dim, 
                  host_data + (idx + 1) * dim, 
                  queries.data() + i * dim);
    }

    {
        std::cout << ">> [GT] Generating Ground Truth using CPU FAISS Flat L2..." << std::endl;
        
        // 1. 创建 CPU Flat L2 索引
        faiss::IndexFlatL2 cpu_index(dim);
        
        // 2. 添加数据
        // host_data 是之前 mmap 读取的 CPU 指针，可以直接传给 FAISS
        // FAISS 内部会处理数据读取
        cpu_index.add(total, host_data);
        
        // 3. 执行搜索
        // queries.data() 是查询向量
        // k 是 Top-K
        // 结果直接写入 gt_dists 和 gt_indices 的 data() 指针
        cpu_index.search(num_queries, queries.data(), k, gt_dists.data(), gt_indices.data());
        
        std::cout << "   GT Generation complete." << std::endl;
    }

    // =========================================================================
    // [分析] 真值分布统计 (Ground Truth Distribution)
    // 统计 queries 的真值到底分布在多少个不同的时间桶里
    // =========================================================================
    std::cout << "\n>> [Analysis] Analyzing Ground Truth Bucket Distribution..." << std::endl;
    
    int total_buckets_hit_sum = 0;
    size_t total_buckets_count = bucket_counts.size();

    // 统计桶的分布
    // 如果一个 Query 的 Top-10 结果来自于 [T1, T1, T2, T3...]，则命中了 3 个桶
    for (int i = 0; i < num_queries; ++i) {
        std::unordered_set<uint64_t> unique_buckets_for_query;
        
        for (int j = 0; j < k; ++j) {
            int64_t gid = gt_indices[i * k + j];
            if (gid >= 0 && gid < (int64_t)timestamps.size()) {
                uint64_t ts = timestamps[gid];
                unique_buckets_for_query.insert(ts);
            }
        }
        total_buckets_hit_sum += (int)unique_buckets_for_query.size();
        std::cout << "   Query " << i << ": Hit " << unique_buckets_for_query.size() << " buckets." << std::endl;
    }

    double avg_buckets_hit = total_buckets_hit_sum / (double)num_queries;
    double avg_coverage_pct = (avg_buckets_hit / total_buckets_count) * 100.0;

    std::cout << "   Avg Buckets Hit per Query: " << std::fixed << std::setprecision(2) 
              << avg_buckets_hit << " / " << total_buckets_count << std::endl;
    std::cout << "   Avg Distribution Coverage: " << avg_coverage_pct << "%" << std::endl;
    
    int probe_buckets_recommend = std::ceil(avg_buckets_hit) + 1; // 推荐探测数：平均命中数 + 1 (Buffer)
    if (probe_buckets_recommend > total_buckets_count) probe_buckets_recommend = total_buckets_count;

    std::cout << "   [Hint] Recommended 'probe_buckets' >= " << probe_buckets_recommend << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    // =========================================================================
    // TimeStampIndex 测试流程
    // =========================================================================
    std::cout << "\n>> [TimeStampIndex] Start Testing..." << std::endl;
    Timer timer;

    // 1. 初始化
    timestamp::TimeStampIndex ts_index(dim, 32, 1000); 

    // 2. 插入数据
    std::cout << ">> Step 1: Inserting " << total << " vectors..." << std::endl;
    timer.reset();
    ts_index.insert(host_data, timestamps.data(), total);
    double insert_time = timer.elapsed_ms();
    std::cout << "   Insert Time: " << insert_time << " ms" << std::endl;

    // 3. 构建虚拟层
    std::cout << ">> Step 2: Building Virtual Index..." << std::endl;
    timer.reset();
    ts_index.build_virtual_index();
    double build_time = timer.elapsed_ms();
    std::cout << "   Virtual Build Time: " << build_time << " ms" << std::endl;

    // 4. 查询
    // 使用基于统计分析的推荐值，或者固定值 (比如 3)
    int probe_buckets = 40;
    if (probe_buckets > total_buckets_count) probe_buckets = total_buckets_count;

    std::cout << ">> Step 3: Querying (Probe " << probe_buckets << " buckets)..." << std::endl;
    
    std::vector<int64_t> out_indices(num_queries * k);
    std::vector<float> out_dists(num_queries * k);

    // 预热
    ts_index.query(queries.data(), 1, out_indices.data(), out_dists.data(), probe_buckets);

    timer.reset();
    for (int i = 0; i < num_queries; ++i) {
        ts_index.query(queries.data() + i * dim, k, 
                       out_indices.data() + i * k, 
                       out_dists.data() + i * k, 
                       probe_buckets);
    }
    double search_time = timer.elapsed_ms();

    // 5. 统计
    int correct_cnt = 0;
    for (int i = 0; i < num_queries; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(gt_indices[i * k + j]);
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(out_indices[i * k + j])) correct_cnt++;
        }
    }
    double recall = 100.0 * correct_cnt / (num_queries * k);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "   Total Vectors: " << ts_index.size() << std::endl;
    std::cout << "   Insert Time:   " << insert_time << " ms" << std::endl;
    std::cout << "   Build Time:    " << build_time << " ms" << std::endl;
    std::cout << "   Search Time:   " << search_time << " ms (" << num_queries << " queries)" << std::endl;
    std::cout << "   Latency:       " << (search_time / num_queries) << " ms/query" << std::endl;
    std::cout << "   QPS:           " << std::fixed << std::setprecision(2) << (num_queries * 1000.0 / search_time) << std::endl;
    std::cout << "   Recall@" << k << ":      " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    munmap((void*)host_data, sz);
    close(fd);

    if (recall > 85.0) {
        std::cout << "PASSED: Recall is acceptable." << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: Recall is too low." << std::endl;
        return 1;
    }
}