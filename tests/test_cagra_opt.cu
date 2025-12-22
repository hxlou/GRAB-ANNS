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

// FAISS (CPU 版本生成 GT)
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

#include "common.cuh"

int main() {
    CHECK_CUDA(cudaSetDevice(CUDA_DEVICE_ID));

    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    std::cout << "==========================================================" << std::endl;
    std::cout << "CagraIndexOpt Integration Test (Partitioned Build)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 2. 加载数据
    int dim = -1, file_total = -1;
    if (!parseMeta(meta_path, dim, file_total)) { std::cerr << "Meta parse failed" << std::endl; return 1; }
    
    // 【限制数据量】50w
    int total = 880000;
    if (file_total < total) total = file_total; 

    std::cout << "Dataset: Using " << total << " vectors, " << dim << " dims." << std::endl;

    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) { std::cerr << "Failed to open bin file" << std::endl; return 1; }
    
    size_t sz = (size_t)total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) { std::cerr << "mmap failed" << std::endl; close(fd); return 1; }

    // 3. 生成时间戳 (构造 10 个桶)
    // 策略：每 5w 条数据一个时间戳 (0, 1, ..., 9)
    // 这样 CagraIndexOpt::build 内部就会识别出 10 个物理分块，分别构建子图
    size_t bucket_num = 40;
    size_t bucket_size = total / bucket_num;
    std::cout << ">> Generating Timestamps (1 bucket = " << bucket_size << " vectors)..." << std::endl;
    
    std::vector<uint64_t> timestamps(total);
    std::map<uint64_t, int> bucket_counts;
    for (int i = 0; i < total; ++i) {
        timestamps[i] = i / bucket_size;
        bucket_counts[timestamps[i]]++;
    }
    std::cout << "   Total Buckets: " << bucket_counts.size() << " (Expected 10)" << std::endl;

    // 4. 准备查询与真值 (Ground Truth)
    const int num_queries = 1000;
    const int k = 10;
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

    // 生成 GT (使用 CPU FAISS，稳健且省显存)
    {
        std::cout << "   Generating GT using CPU FAISS..." << std::endl;
        faiss::IndexFlatL2 cpu_index(dim);
        cpu_index.add(total, host_data); // 零拷贝
        cpu_index.search(num_queries, queries.data(), k, gt_dists.data(), gt_indices.data());
    }

    // =========================================================================
    // CagraIndexOpt 测试流程
    // =========================================================================
    std::cout << "\n>> [CagraIndexOpt] Start Testing..." << std::endl;
    Timer timer;

    // 1. 初始化
    // dim=1024, degree=32, VMM=default
    cagra::CagraIndexOpt index(dim); 

    // 2. 添加数据 (Host RAM)
    std::cout << ">> Step 1: Adding data..." << std::endl;
    timer.reset();
    index.add(total, host_data, timestamps.data());
    double add_time = timer.elapsed_ms();
    std::cout << "   Add Time: " << add_time << " ms" << std::endl;

    // 3. 构建索引 (GPU Build)
    // 这里会触发 build_time_partitioned_graph -> Local Build -> Global Remote Edge
    std::cout << ">> Step 2: Building Index (Partitioned + Global Connect)..." << std::endl;
    timer.reset();
    index.build();
    double build_time = timer.elapsed_ms();
    std::cout << "   Build Time: " << build_time << " ms" << std::endl;

    // 4. 查询 (全量搜索，不带时间过滤)
    // 验证 Global Edges 是否有效地连接了各个时间桶
    std::cout << ">> Step 3: Querying (Global Search)..." << std::endl;
    
    std::vector<int64_t> out_indices(num_queries * k);
    std::vector<float> out_dists(num_queries * k);

    // 设置搜索参数
    index.setQueryParams(
        512,  // itopk
        6,    // search_width (初始设为1，如果Recall低可以调大)
        0,    // min_iter
        100,   // max_iter (跨桶搜索需要更多步数)
        16    // hash_bitlen
    );

    // 预热
    index.query(queries.data(), 1, k, 0, UINT64_MAX, out_indices.data(), out_dists.data());

    timer.reset();
    index.query(queries.data(), num_queries, k, 
                0, UINT64_MAX, // 全时间范围
                out_indices.data(), out_dists.data());
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
    std::cout << "   Total Vectors: " << index.size() << std::endl;
    std::cout << "   Build Time:    " << build_time << " ms" << std::endl;
    std::cout << "   Search Time:   " << search_time << " ms" << std::endl;
    std::cout << "   QPS:           " << std::fixed << std::setprecision(2) << (num_queries * 1000.0 / search_time) << std::endl;
    std::cout << "   Recall@" << k << ":      " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    munmap((void*)host_data, sz);
    close(fd);

    // 验证逻辑：
    // 如果 Recall 很高 (>95%)，说明 Global Remote Edges 工作正常，成功连接了孤立的桶。
    // 如果 Recall 很低 (~10%)，说明搜索被困在了 Query 所在的那个时间桶里，跨桶失败。
    if (recall > 90.0) {
        std::cout << "PASSED: Graph is well-connected." << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: Low recall, graph might be partitioned." << std::endl;
        return 1;
    }
}