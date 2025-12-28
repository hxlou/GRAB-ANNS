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
#include <map>
#include <numeric>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <faiss/IndexFlat.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

// =============================================================================
// 1. 配置结构体定义
// =============================================================================

// 构建参数组合
struct BuildConfig {
    size_t dataset_size;    // 使用多少数据 (e.g. 50w, 88w)
    size_t num_buckets;     // 分多少个桶 (e.g. 10, 40)
    uint32_t graph_degree;  // 构图度数 (e.g. 32, 64)
    
    std::string to_string() const {
        return "Data=" + std::to_string(dataset_size) + 
               ", Buckets=" + std::to_string(num_buckets) + 
               ", Degree=" + std::to_string(graph_degree);
    }
};

// 搜索参数组合
struct SearchConfig {
    uint32_t itopk;         // 内部候选池
    uint32_t width;         // 搜索宽度
    uint32_t iter;          // 最大迭代
    
    std::string to_string() const {
        return "Itopk=" + std::to_string(itopk) + 
               ", Width=" + std::to_string(width) + 
               ", Iter=" + std::to_string(iter);
    }
};

// =============================================================================
// 2. 辅助工具
// =============================================================================

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

bool parseMeta(const std::string& path, int& dim, int& total) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("\"dim\"") != std::string::npos) dim = std::stoi(line.substr(line.find(":") + 1));
        if (line.find("\"count\"") != std::string::npos) total = std::stoi(line.substr(line.find(":") + 1));
    }
    if (dim <= 0) dim = 1024;
    return true;
}

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

// CSV Logger
void log_csv(const BuildConfig& b_conf, const SearchConfig& s_conf, 
             double build_time, double g_recall, double g_qps, double l_recall, double l_qps) {
    std::ofstream file("benchmark_results.csv", std::ios::app);
    if (file.tellp() == 0) {
        file << "Dataset,Buckets,Degree,BuildTime(ms),Itopk,Width,Iter,GlobalRecall,GlobalQPS,LocalRecall,LocalQPS\n";
    }
    file << b_conf.dataset_size << "," << b_conf.num_buckets << "," << b_conf.graph_degree << "," 
         << build_time << ","
         << s_conf.itopk << "," << s_conf.width << "," << s_conf.iter << ","
         << g_recall << "," << g_qps << "," << l_recall << "," << l_qps << "\n";
}

// =============================================================================
// 3. 核心测试函数 (隔离的评估逻辑)
// =============================================================================
// 这个函数不关心索引是怎么构建的，它只负责用给定的 search_conf 去跑测试
void evaluate_configuration(
    cagra::CagraIndexOpt& index,        // 已构建好的索引
    const float* host_data_full,        // 原始全量数据 (用于生成 GT)
    const std::vector<uint64_t>& timestamps,// 时间戳
    int dim,
    const BuildConfig& b_conf,
    const SearchConfig& s_conf,
    double build_time_ms
) {
    // 设置搜索参数
    index.setQueryParams(s_conf.itopk, s_conf.width, 0, s_conf.iter, 14);

    const int NUM_QUERIES = 100;
    const int K = 10;
    Timer timer;

    // -------------------------------------------------------
    // A. 全量搜索测试 (Global Search)
    // -------------------------------------------------------
    std::vector<float> queries(NUM_QUERIES * dim);
    std::vector<int64_t> gt_global(NUM_QUERIES * K);
    std::vector<float> gt_dists(NUM_QUERIES * K);
    
    // 随机采样 Query
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, b_conf.dataset_size - 1);
    for(int i=0; i<NUM_QUERIES; ++i) {
        int idx = dist(rng);
        std::copy(host_data_full + idx * dim, host_data_full + (idx + 1) * dim, queries.data() + i * dim);
    }

    // 生成 Global GT (CPU)
    {
        faiss::IndexFlatL2 cpu_index(dim);
        cpu_index.add(b_conf.dataset_size, host_data_full);
        cpu_index.search(NUM_QUERIES, queries.data(), K, gt_dists.data(), gt_global.data());
    }

    // 执行 CAGRA Global Query
    std::vector<int64_t> out_ids(NUM_QUERIES * K);
    std::vector<float> out_ds(NUM_QUERIES * K);
    
    // 预热
    index.query(queries.data(), 1, K, 0, UINT64_MAX, out_ids.data(), out_ds.data());
    
    timer.reset();
    index.query(queries.data(), NUM_QUERIES, K, 0, UINT64_MAX, out_ids.data(), out_ds.data());
    double g_time = timer.elapsed_ms();
    
    double g_recall = calc_recall(NUM_QUERIES, K, gt_global.data(), out_ids.data());
    double g_qps = NUM_QUERIES * 1000.0 / g_time;

    // -------------------------------------------------------
    // B. 局部搜索测试 (Local Search)
    // -------------------------------------------------------
    // 随机选 3 个桶进行测试，取平均值
    std::vector<int> test_buckets;
    std::uniform_int_distribution<int> b_dist(0, b_conf.num_buckets - 1);
    for(int i=0; i<3; ++i) test_buckets.push_back(b_dist(rng));

    double total_l_recall = 0;
    double total_l_time = 0;

    for (int ts : test_buckets) {
        // 获取该桶的真实数据范围 (通过 Index 接口获取 ID 列表)
        std::vector<uint32_t> bucket_ids = index.get_ids_by_timestamp(ts);
        size_t b_size = bucket_ids.size();
        if (b_size == 0) continue;

        // Gather 数据
        std::vector<float> bucket_vecs(b_size * dim);
        for(size_t i=0; i<b_size; ++i) {
            const float* src = index.get_data() + (size_t)bucket_ids[i] * dim;
            std::copy(src, src+dim, bucket_vecs.data() + i*dim);
        }

        // 生成 Local Query
        std::vector<float> l_queries(NUM_QUERIES * dim);
        std::uniform_int_distribution<int> l_dist(0, b_size - 1);
        for(int i=0; i<NUM_QUERIES; ++i) {
            int idx = l_dist(rng);
            std::copy(bucket_vecs.data() + idx * dim, bucket_vecs.data() + (idx + 1) * dim, l_queries.data() + i * dim);
        }

        // 生成 Local GT
        std::vector<int64_t> l_gt_local_idx(NUM_QUERIES * K);
        std::vector<float> l_gt_dists(NUM_QUERIES * K);
        {
            faiss::IndexFlatL2 sub_index(dim);
            sub_index.add(b_size, bucket_vecs.data());
            sub_index.search(NUM_QUERIES, l_queries.data(), K, l_gt_dists.data(), l_gt_local_idx.data());
        }
        
        // 转换 GT: Local -> Global
        std::vector<int64_t> l_gt_global_idx(NUM_QUERIES * K);
        for(size_t i=0; i<l_gt_global_idx.size(); ++i) {
            l_gt_global_idx[i] = bucket_ids[l_gt_local_idx[i]];
        }

        // CAGRA Local Query
        timer.reset();
        index.query_local(l_queries.data(), NUM_QUERIES, K, ts, out_ids.data(), out_ds.data(), 0);
        total_l_time += timer.elapsed_ms();

        total_l_recall += calc_recall(NUM_QUERIES, K, l_gt_global_idx.data(), out_ids.data());
    }
    
    double l_recall = total_l_recall / test_buckets.size();
    double l_qps = (NUM_QUERIES * test_buckets.size()) * 1000.0 / total_l_time;

    // -------------------------------------------------------
    // C. 输出与记录
    // -------------------------------------------------------
    std::cout << "   [Result] " << s_conf.to_string() 
              << " | G-Recall: " << std::fixed << std::setprecision(2) << g_recall << "% (" << (int)g_qps << " QPS)"
              << " | L-Recall: " << l_recall << "% (" << (int)l_qps << " QPS)" << std::endl;

    log_csv(b_conf, s_conf, build_time_ms, g_recall, g_qps, l_recall, l_qps);
}


// =============================================================================
// Main: 参数定义与调度
// =============================================================================
int main() {
    int cuda_device = 0; 
    CHECK_CUDA(cudaSetDevice(cuda_device));

    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    // 1. 加载元数据
    int dim = 1024, file_total = 0;
    if (!parseMeta(meta_path, dim, file_total)) return 1;

    // 2. 加载全量数据 (Host)
    int fd = open(bin_path.c_str(), O_RDONLY);
    size_t file_sz = (size_t)file_total * dim * sizeof(float);
    const float* host_full_data = (const float*)mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    
    // ============================================================
    // 【参数定义区】在这里写死你需要测试的所有组合
    // ============================================================
    
    // A. 构建参数列表 (外层循环)
    std::vector<BuildConfig> build_configs = {
        // {DataSize, Buckets, Degree, LocalDegree}
        {880000, 2, 32},
        {880000, 2, 64},
        {880000, 2, 128},
        {880000, 10, 32},
        {880000, 10, 64},
        {880000, 10, 128},
        {880000, 50, 32},
        {880000, 50, 64},
        {880000, 50, 128},
        {880000, 100, 32},
        {880000, 100, 64},
        {880000, 100, 128},
        {880000, 500, 32},
        {880000, 500, 64},
        {880000, 500, 32}
    };

    // B. 搜索参数列表 (内层循环)
    std::vector<SearchConfig> search_configs = {
        // {Itopk, Width, Iter}
        // {128, 4, 50}, // 极速模式
        // {128, 6, 100}, // 平衡模式
        // {256, 4, 100}, // 高召回模式
        // {256, 4, 50 },
        // {256, 6, 50 },
        // {256, 6, 100},
        // {512, 8, 100}  // 极限模式
        {512, 4, 100},
        {512, 4, 200},
        {512, 6, 100},
        {512, 6, 200}
    };

    std::cout << "==========================================================" << std::endl;
    std::cout << "Automated Benchmark Runner" << std::endl;
    std::cout << "Tasks: " << build_configs.size() << " Builds x " << search_configs.size() << " Searches" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ============================================================
    // 调度循环
    // ============================================================
    for (const auto& b_conf : build_configs) {
        std::cout << "\n>>> [Build Config] " << b_conf.to_string() << std::endl;

        // 1. 准备该 Build Config 所需的数据切片和时间戳
        // 截取前 dataset_size 个数据
        size_t current_total = b_conf.dataset_size;
        if (current_total > file_total) current_total = file_total;

        std::vector<uint64_t> timestamps(current_total);
        size_t bucket_size = current_total / b_conf.num_buckets;
        if (bucket_size == 0) bucket_size = 1;

        for(size_t i=0; i<current_total; ++i) {
            timestamps[i] = i / bucket_size;
        }

        // 2. 构建索引 (Expensive)
        cagra::CagraIndexOpt index(dim, b_conf.graph_degree);
        // KNN 构建参数通常设为图度数的 2 倍
        index.setBuildParams(b_conf.graph_degree * 2, b_conf.graph_degree);
        
        Timer build_timer;
        index.add(current_total, host_full_data, timestamps.data());
        index.build();
        double build_time = build_timer.elapsed_ms();
        std::cout << "   Index Built in " << build_time << " ms." << std::endl;

        // 3. 遍历搜索参数 (Cheap)
        for (const auto& s_conf : search_configs) {
            evaluate_configuration(index, host_full_data, timestamps, dim, b_conf, s_conf, build_time);
        }
    }

    std::cout << "\n>>> All benchmarks completed. Results saved to benchmark_results.csv" << std::endl;

    munmap((void*)host_full_data, file_sz);
    close(fd);
    return 0;
}