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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <faiss/IndexFlat.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

// =============================================================================
// 1. 配置结构体
// =============================================================================

struct BuildConfig {
    size_t total_data_size;   // 总数据量
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

// 加载 SIFT .fvecs 格式
void load_fvecs(const std::string& filename, std::vector<float>& data, int& dim, size_t& num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        exit(1);
    }
    in.read((char*)&dim, sizeof(int));
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t row_size = sizeof(int) + dim * sizeof(float);
    num = file_size / row_size;

    std::cout << ">> Loading SIFT from " << filename << " (Dim=" << dim << ", N=" << num << ")..." << std::endl;
    data.resize(num * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        int d;
        in.read((char*)&d, sizeof(int));
        if (d != dim) { std::cerr << "Dim mismatch!" << std::endl; exit(1); }
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
}

// 任务包：一个构建配置 + 一组搜索配置
struct BenchmarkTask {
    BuildConfig build_conf;
    std::vector<SearchConfig> search_confs;
};

struct ResultStats {
    double insert_time_ms; // 构建+插入的总耗时 (作为参考)
    double global_recall;
    double global_qps;
    double local_recall;
    double local_qps;
    int local_bound_errors;
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

void log_csv(const std::string& filename, const BuildConfig& b, const SearchConfig& s, const ResultStats& r) {
    std::ofstream file(filename, std::ios::app);
    if (file.tellp() == 0) {
        file << "Dataset,Buckets,Degree,SearchK,Width,Iter,InsertTime(ms),G_Recall,G_QPS,L_Recall,L_QPS,L_Errors\n";
    }
    file << b.total_data_size << "," << b.num_buckets << "," << b.graph_degree << ","
         << s.itopk << "," << s.width << "," << s.iter << ","
         << r.insert_time_ms << ","
         << r.global_recall << "," << r.global_qps << "," 
         << r.local_recall << "," << r.local_qps << "," << r.local_bound_errors << "\n";
}

// =============================================================================
// 核心：执行一组测试
// =============================================================================
void run_task(const float* host_raw_data, int dim, const BenchmarkTask& task, const std::string& csv_file) {
    const auto& b_conf = task.build_conf;
    std::cout << "\n>>> [BUILD PHASE] Config: " << b_conf.to_string() << std::endl;

    // -------------------------------------------------------------
    // 1. 准备数据与切分 (50% Build, 50% Insert)
    // -------------------------------------------------------------
    size_t bucket_size = b_conf.total_data_size / b_conf.num_buckets;
    std::vector<uint64_t> timestamps(b_conf.total_data_size);
    
    std::vector<float> build_data;
    std::vector<uint64_t> build_ts;
    std::vector<float> insert_data;
    std::vector<uint64_t> insert_ts;
    
    build_data.reserve(b_conf.total_data_size/2 * dim);
    insert_data.reserve(b_conf.total_data_size/2 * dim);

    for(size_t i=0; i<b_conf.total_data_size; ++i) {
        uint64_t ts = i / bucket_size;
        timestamps[i] = ts;
        const float* vec = host_raw_data + i * dim;

        // 每个桶的前50%用于Build，后50%用于Insert
        if ((i % bucket_size) < (bucket_size / 2)) {
            build_data.insert(build_data.end(), vec, vec + dim);
            build_ts.push_back(ts);
        } else {
            insert_data.insert(insert_data.end(), vec, vec + dim);
            insert_ts.push_back(ts);
        }
    }

    // -------------------------------------------------------------
    // 2. 初始化与构建 (一次性昂贵操作)
    // -------------------------------------------------------------
    cagra::CagraIndexOpt index(dim, b_conf.graph_degree);
    index.setBuildParams(b_conf.graph_degree * 2, b_conf.graph_degree);
    
    // 设置一个默认的 Search Params 供 Insert 内部使用
    // Insert 阶段为了图质量，建议使用较激进的参数
    index.setQueryParams(256, 4, 0, 50, 14);

    // Phase 1: Build
    std::cout << "   Building Base Index (" << build_ts.size() << " vecs)..." << std::flush;
    index.add(build_ts.size(), build_data.data(), build_ts.data());
    index.build();
    std::cout << " Done." << std::endl;

    // Phase 2: Insert
    std::cout << "   Inserting Incremental Data (" << insert_ts.size() << " vecs)..." << std::flush;
    Timer timer;
    index.insert(insert_ts.size(), insert_data.data(), insert_ts.data());
    double insert_time = timer.elapsed_ms();
    std::cout << " Done. Time: " << insert_time << " ms" << std::endl;

    // -------------------------------------------------------------
    // 3. 准备全局真值 (Global GT) - 只需做一次
    // -------------------------------------------------------------
    const int NUM_Q = 100;
    const int K = 10;
    std::mt19937 rng(42);
    
    const float* index_data_ptr = index.get_data(); // 获取 Index 内部整理好的全量数据
    size_t total_vecs_in_index = index.size();

    std::vector<float> global_queries(NUM_Q * dim);
    std::vector<int64_t> gt_global_idx(NUM_Q * K);
    std::vector<float> gt_global_d(NUM_Q * K);

    // 采样 Query
    std::uniform_int_distribution<size_t> dist(0, total_vecs_in_index - 1);
    for(int i=0; i<NUM_Q; ++i) {
        size_t idx = dist(rng);
        std::copy(index_data_ptr + idx * dim, index_data_ptr + (idx + 1) * dim, global_queries.data() + i * dim);
    }

    // 生成 Global GT
    std::cout << "   Generating Global GT (CPU FAISS)..." << std::flush;
    {
        faiss::IndexFlatL2 gt_index(dim);
        gt_index.add(total_vecs_in_index, index_data_ptr);
        gt_index.search(NUM_Q, global_queries.data(), K, gt_global_d.data(), gt_global_idx.data());
    }
    std::cout << " Done." << std::endl;

    // -------------------------------------------------------------
    // 4. 循环测试 Search Configs (Cheap)
    // -------------------------------------------------------------
    std::cout << ">> [SEARCH PHASE] Testing " << task.search_confs.size() << " configurations..." << std::endl;

    for (const auto& s_conf : task.search_confs) {
        // 更新搜索参数
        // Hash Bitlen 固定 14 (16KB SMEM), 足够一般场景
        index.setQueryParams(s_conf.itopk, s_conf.width, 0, s_conf.iter, 16);

        ResultStats stats;
        stats.insert_time_ms = insert_time;

        // --- A. Global Search Test ---
        {
            std::vector<int64_t> out_idx(NUM_Q * K);
            std::vector<float> out_d(NUM_Q * K);
            
            timer.reset();
            index.query(global_queries.data(), NUM_Q, K, 0, UINT64_MAX, out_idx.data(), out_d.data());
            double t = timer.elapsed_ms();

            stats.global_recall = calc_recall(NUM_Q, K, gt_global_idx.data(), out_idx.data());
            stats.global_qps = NUM_Q * 1000.0 / t;
        }

        // --- B. Local Search Test ---
        {
            // 随机选 3 个桶进行测试
            std::vector<int> test_buckets;
            std::uniform_int_distribution<int> b_dist(0, b_conf.num_buckets - 1);
            for(int i=0; i<3; ++i) test_buckets.push_back(b_dist(rng));

            double sum_recall = 0;
            double sum_time = 0;
            int total_local_q = 0;

            for (int ts : test_buckets) {
                // Gather Data & GT Generation (针对每个桶动态生成)
                std::vector<uint32_t> bucket_gids = index.get_ids_by_timestamp(ts);
                size_t b_size = bucket_gids.size();
                if(b_size == 0) continue;

                std::vector<float> bucket_vecs(b_size * dim);
                for(size_t i=0; i<b_size; ++i) {
                    const float* src = index_data_ptr + (size_t)bucket_gids[i] * dim;
                    std::copy(src, src+dim, bucket_vecs.data() + i*dim);
                }

                // Local Query
                std::vector<float> l_queries(NUM_Q * dim);
                std::uniform_int_distribution<int> l_dist(0, b_size - 1);
                for(int i=0; i<NUM_Q; ++i) {
                    int local_idx = l_dist(rng);
                    std::copy(bucket_vecs.data() + local_idx*dim, bucket_vecs.data() + (local_idx+1)*dim, l_queries.data() + i*dim);
                }

                // Local GT
                std::vector<int64_t> gt_local_raw(NUM_Q * K); 
                std::vector<float> gt_d(NUM_Q * K);
                faiss::IndexFlatL2 local_gt_index(dim);
                local_gt_index.add(b_size, bucket_vecs.data());
                local_gt_index.search(NUM_Q, l_queries.data(), K, gt_d.data(), gt_local_raw.data());

                std::vector<int64_t> gt_global(NUM_Q * K);
                for(size_t i=0; i<gt_global.size(); ++i) gt_global[i] = bucket_gids[gt_local_raw[i]];

                // Run Search
                std::vector<int64_t> out_idx(NUM_Q * K);
                std::vector<float> out_d(NUM_Q * K);
                
                timer.reset();
                index.query_local(l_queries.data(), NUM_Q, K, ts, out_idx.data(), out_d.data(), b_conf.graph_degree - 4);
                sum_time += timer.elapsed_ms();

                // Validate
                std::unordered_set<uint32_t> valid_set(bucket_gids.begin(), bucket_gids.end());
                for(auto gid : out_idx) {
                    if (gid != -1 && valid_set.find((uint32_t)gid) == valid_set.end()) {
                        stats.local_bound_errors++;
                    }
                }
                
                sum_recall += calc_recall(NUM_Q, K, gt_global.data(), out_idx.data());
                total_local_q += NUM_Q;
            }
            
            if (total_local_q > 0) {
                stats.local_recall = sum_recall / test_buckets.size();
                stats.local_qps = total_local_q * 1000.0 / sum_time;
            }
        }

        // Log Result
        std::cout << "   " << s_conf.to_string() 
                  << " | G-Rec: " << std::fixed << std::setprecision(2) << stats.global_recall << "%"
                  << " | L-Rec: " << stats.local_recall << "%" << std::endl;

        log_csv(csv_file, b_conf, s_conf, stats);
    }
}

// =============================================================================
// Main
// =============================================================================
int main() {
    int cuda_device = 0; 
    CHECK_CUDA(cudaSetDevice(cuda_device));

    // std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    // std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";
    std::string fvecs_path = "../data/sift-1m/sift/sift_base.fvecs";
    std::string csv_file = "benchmark_insert_sift.csv";

    // 加载全量数据
    int dim = 1024;
    size_t file_total = 0;
    std::vector<float> host_full_data_vec;
    load_fvecs(fvecs_path, host_full_data_vec, dim, file_total);

    // ==========================================================
    // 定义测试任务 (Build Configs + Search Configs)
    // ==========================================================
    std::vector<BenchmarkTask> tasks;

    // Task 1: 50w数据, 10个桶, Degree 32
    // tasks.push_back({
    //     {880000, 2, 32}, // Build Config
    //     { // Search Configs to test on this index
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // tasks.push_back({
    //     {880000, 2, 64}, // Build Config
    //     { // Search Configs to test on this index
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // tasks.push_back({
    //     {880000, 2, 128}, // Build Config
    //     { // Search Configs to test on this index
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}

    //     }
    // });

    // // Task 2: 50w数据, 40个桶, Degree 32 (更细碎的桶，测试连通性)
    // tasks.push_back({
    //     {880000, 10, 32},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // tasks.push_back({
    //     {880000, 10, 64},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // tasks.push_back({
    //     {880000, 10, 128},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });


    // // Task 3: 全量88w数据, 80个桶 (压力测试)
    // tasks.push_back({
    //     {880000, 50, 32},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // tasks.push_back({
    //     {880000, 50, 64},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // tasks.push_back({
    //     {880000, 50, 128},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    // // Task 4: 全量88w数据, 80个桶 (压力测试)
    // tasks.push_back({
    //     {1000000, 100, 32},
    //     {
    //         {128, 4, 50},
    //         {128, 4, 100},
    //         {256, 4, 50},
    //         {256, 4, 100},
    //         {512, 4, 100},
    //         {512, 4, 200}
    //     }
    // });

    tasks.push_back({
        {1000000, 100, 64},
        {
            {128, 4, 50},
            {128, 4, 100},
            {256, 4, 50},
            {256, 4, 100},
            {512, 4, 100},
            {512, 4, 200}
        }
    });

    tasks.push_back({
        {1000000, 100, 128},
        {
            {128, 4, 50},
            {128, 4, 100},
            {256, 4, 50},
            {256, 4, 100},
            {512, 4, 100},
            {512, 4, 200}
        }
    });

    std::cout << "Starting Optimized Benchmark (" << tasks.size() << " Build Tasks)..." << std::endl;

    for (const auto& task : tasks) {
        // 确保数据量不超限
        if (task.build_conf.total_data_size <= file_total) {
            run_task(host_full_data_vec.data(), dim, task, csv_file);
        }
    }

    return 0;
}