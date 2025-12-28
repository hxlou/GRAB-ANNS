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

// 系统库
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS (CPU 版本)
#include <faiss/IndexFlat.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

// =============================================================================
// 1. 配置结构体
// =============================================================================
struct BuildConfig {
    size_t total_data_size;   // 数据量
    size_t num_buckets;       // 桶数量
    uint32_t graph_degree;    // 图度数
    
    std::string to_string() const {
        return "N=" + std::to_string(total_data_size) + 
               " Bkt=" + std::to_string(num_buckets) + 
               " Deg=" + std::to_string(graph_degree);
    }
};

struct SearchConfig {
    uint32_t itopk;
    uint32_t width;
    uint32_t iter;
    
    std::string to_string() const {
        return "[k=" + std::to_string(itopk) + 
               " w=" + std::to_string(width) + 
               " i=" + std::to_string(iter) + "]";
    }
};

struct RangeStats {
    double ratio = 0.0;           
    double avg_recall = 0.0;
    double avg_qps = 0.0;
    long long bound_errors = 0;
};

// =============================================================================
// 2. 辅助工具 & SIFT 加载器
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

// SIFT .fvecs 读取函数
// 格式: [int32_t dim] [float v1] [float v2] ... [float vd] 重复 N 次
// 我们需要去掉 dim 头，把 float 数据紧凑存入 vector
void load_fvecs(const std::string& filename, std::vector<float>& data, int& dim, size_t& num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        exit(1);
    }

    // 读取第一个 dim
    in.read((char*)&dim, sizeof(int));
    if (dim <= 0 || dim > 2048) {
        std::cerr << "Invalid dimension in fvecs: " << dim << std::endl;
        exit(1);
    }

    // 计算文件大小推导向量数量
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t row_size = sizeof(int) + dim * sizeof(float);
    num = file_size / row_size;

    std::cout << ">> Loading SIFT-1M from " << filename << "..." << std::endl;
    std::cout << "   Dim: " << dim << ", Count: " << num << std::endl;

    // 预分配内存
    data.resize(num * dim);

    // 回到开头循环读取
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        int d;
        in.read((char*)&d, sizeof(int)); // 跳过 header
        if (d != dim) {
            std::cerr << "Dimension mismatch at vector " << i << std::endl;
            exit(1);
        }
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
    std::cout << "   Load complete." << std::endl;
}

double calc_recall(size_t nq, int k, const int64_t* gt, const int64_t* res) {
    size_t correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(gt[i * k + j]);
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(res[i * k + j])) correct++;
        }
    }
    return 100.0 * (double)correct / (nq * k);
}

void log_csv(const std::string& filename, const BuildConfig& b, const SearchConfig& s, double build_time, const RangeStats& r) {
    std::ofstream file(filename, std::ios::app);
    if (file.tellp() == 0) {
        file << "Dataset,Buckets,Degree,BuildTime(ms),Itopk,Width,Iter,RangeRatio,Recall,QPS,BoundErrors\n";
    }
    file << b.total_data_size << "," << b.num_buckets << "," << b.graph_degree << "," << build_time << ","
         << s.itopk << "," << s.width << "," << s.iter << ","
         << r.ratio << "," << r.avg_recall << "," << r.avg_qps << "," << r.bound_errors << "\n";
}

// =============================================================================
// 3. 核心测试逻辑 (Range Search Benchmark)
// =============================================================================
void run_range_benchmark(
    cagra::CagraIndexOpt& index,
    const float* host_data,
    int dim,
    const BuildConfig& b_conf,
    const SearchConfig& s_conf,
    double build_time,
    const std::vector<double>& ratios,
    const std::string& csv_file
) {
    // 设置搜索参数
    index.setQueryParams(s_conf.itopk, s_conf.width, 0, s_conf.iter, 14);

    const int NUM_ROUNDS = 5;       
    const int QUERIES_PER_ROUND = 32;
    const int K = 10;
    
    std::mt19937 rng(12345);
    size_t bucket_size = b_conf.total_data_size / b_conf.num_buckets;

    std::cout << "   [Search Config] " << s_conf.to_string() << std::endl;

    for (double ratio : ratios) {
        RangeStats stats = {}; // 零初始化
        stats.ratio = ratio;
        
        int span_buckets = (int)(b_conf.num_buckets * ratio);
        if (span_buckets < 1) span_buckets = 1;
        if (span_buckets > b_conf.num_buckets) span_buckets = b_conf.num_buckets;

        // 多轮测试取平均
        for (int round = 0; round < NUM_ROUNDS; ++round) {
            // A. 随机生成范围 [start, end)
            int max_start = b_conf.num_buckets - span_buckets;
            int start_bucket = std::uniform_int_distribution<int>(0, max_start)(rng);
            int end_bucket = start_bucket + span_buckets;

            size_t range_global_start = start_bucket * bucket_size;
            size_t range_global_end = std::min((size_t)end_bucket * bucket_size, b_conf.total_data_size);
            size_t range_len = range_global_end - range_global_start;
            const float* range_data_ptr = host_data + range_global_start * dim;

            // B. 随机 Query (从范围内的 Base Data 中采样)
            std::vector<float> queries(QUERIES_PER_ROUND * dim);
            std::uniform_int_distribution<int> q_dist(0, range_len - 1);
            
            for (int i = 0; i < QUERIES_PER_ROUND; ++i) {
                int local_idx = q_dist(rng);
                std::copy(range_data_ptr + local_idx * dim,
                          range_data_ptr + (local_idx + 1) * dim,
                          queries.data() + i * dim);
            }

            // C. 动态生成 Range GT (CPU FAISS)
            std::vector<int64_t> gt_indices(QUERIES_PER_ROUND * K);
            std::vector<float> gt_dists(QUERIES_PER_ROUND * K);
            {
                faiss::IndexFlatL2 cpu_index(dim);
                cpu_index.add(range_len, range_data_ptr);
                cpu_index.search(QUERIES_PER_ROUND, queries.data(), K, gt_dists.data(), gt_indices.data());
            }
            
            // D. CAGRA Range Search
            std::vector<int64_t> out_indices(QUERIES_PER_ROUND * K);
            std::vector<float> out_dists(QUERIES_PER_ROUND * K);
            
            Timer t;
            
            // 确保 active_degree 传入 graph_degree (32)，启用 Remote Edge
            index.query_range(queries.data(), QUERIES_PER_ROUND, K, 
                              (uint64_t)start_bucket, (uint64_t)end_bucket, 
                              out_indices.data(), out_dists.data(), b_conf.graph_degree);
            
            double ms = t.elapsed_ms();

            // E. 统计
            // 越界检查
            for (int i = 0; i < QUERIES_PER_ROUND * K; ++i) {
                int64_t gid = out_indices[i];
                if (gid != -1) {
                    if (gid < range_global_start || gid >= range_global_end) {
                        stats.bound_errors++;
                    }
                }
            }

            // 转换 GT 到 Global ID
            std::vector<int64_t> gt_global(QUERIES_PER_ROUND * K);
            for(int i=0; i<QUERIES_PER_ROUND * K; ++i) gt_global[i] = gt_indices[i] + range_global_start;

            stats.avg_recall += calc_recall(QUERIES_PER_ROUND, K, gt_global.data(), out_indices.data());
            stats.avg_qps += QUERIES_PER_ROUND * 1000.0 / ms;
        }

        stats.avg_recall /= NUM_ROUNDS;
        stats.avg_qps /= NUM_ROUNDS;

        std::cout << "      Ratio=" << (int)(ratio*100) << "% | Recall=" 
                  << std::fixed << std::setprecision(2) << stats.avg_recall << "% | QPS=" 
                  << (int)stats.avg_qps << " | Errors=" << stats.bound_errors << std::endl;
        
        log_csv(csv_file, b_conf, s_conf, build_time, stats);
    }
}


// =============================================================================
// Main
// =============================================================================
int main() {
    int cuda_device = 0; 
    CHECK_CUDA(cudaSetDevice(cuda_device));

    std::string sift_path = "../data/sift-1m/sift/sift_base.fvecs";
    std::string csv_file  = "benchmark_range_sift.csv";

    // 1. 加载 SIFT 数据
    std::vector<float> host_full_data;
    int dim = 0;
    size_t file_total = 0;
    
    // 使用新的 Loader
    load_fvecs(sift_path, host_full_data, dim, file_total);
    
    // 注意：SIFT 是 128 维，不是 1024。
    // 我们之前的 CagraIndexOpt 默认是 1024 维。如果你的 config.cuh 是硬编码的 1024，需要修改！
    // 或者在 CagraIndexOpt 构造函数里做 check。
    // 如果 config.cuh 是 constexpr DIM=1024，那么这个测试跑 SIFT(128) 可能会有问题（读取越界）。
    // **假设你已经将 config.cuh 中的 DIM 改为了 128，或者 SIFT 数据 padding 到了 1024**
    // (为了简单起见，这里假设你的 config.cuh 里的 DIM 已经适配了 SIFT 的 128 维)

    if (dim != 960) {
        std::cerr << "Warning: GIST should be 960 dim, but got " << dim << std::endl;
    }

    // ==========================================================
    // 参数配置区
    // ==========================================================
    
    // A. Build 参数
    std::vector<BuildConfig> build_configs = {
        // SIFT-1M base 有 100w 数据
        // 我们测试 100w，分 100 个桶 (每个 1w)
        // {1000000, 100, 32},
        {1000000, 100, 64},
        {1000000, 100, 128},
        // 或者分 20 个桶 (每个 5w)
        // {1000000, 20, 32},
        {1000000, 20, 64},
        {1000000, 20, 128}
    };

    // B. Search 参数
    std::vector<SearchConfig> search_configs = {
        // {Itopk, Width, Iter}
        {128, 4, 50},
        {128, 4, 100},
        {128, 4, 150},
        {128, 4, 200},
        {256, 4, 50},
        {256, 4, 100},
        {256, 4, 150},
        {256, 4, 200},  
        {512, 4, 50},
        {512, 4, 100},
        {512, 4, 150},
        {512, 4, 200}
    };

    // C. 测试范围比例
    std::vector<double> test_ratios = {0.2, 0.4, 0.6, 0.8};

    std::cout << "Starting Range Benchmark on SIFT-1M..." << std::endl;

    for (const auto& b_conf : build_configs) {
        if (b_conf.total_data_size > file_total) {
            std::cout << "Skipping config (data size > file size)" << std::endl;
            continue;
        }
        
        std::cout << "\n>>> [Build] " << b_conf.to_string() << std::endl;

        // 生成时间戳
        size_t bucket_size = b_conf.total_data_size / b_conf.num_buckets;
        std::vector<uint64_t> timestamps(b_conf.total_data_size);
        for(int i=0; i<b_conf.total_data_size; ++i) timestamps[i] = i / bucket_size;

        // 构建索引
        // 注意：这里的 dim 需要传入 SIFT 的 128
        cagra::CagraIndexOpt index(dim, b_conf.graph_degree);
        index.setBuildParams(b_conf.graph_degree * 2, b_conf.graph_degree);
        
        Timer timer;
        // host_full_data 是 vector，用 .data()
        index.add(b_conf.total_data_size, host_full_data.data(), timestamps.data());
        index.build();
        double build_time = timer.elapsed_ms();
        std::cout << "    Build Time: " << build_time << " ms" << std::endl;

        // 遍历搜索参数
        for (const auto& s_conf : search_configs) {
            run_range_benchmark(index, host_full_data.data(), dim, b_conf, s_conf, build_time, test_ratios, csv_file);
        }
    }

    return 0;
}