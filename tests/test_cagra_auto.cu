#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <set>
#include <iomanip>
#include <omp.h> // 务必开启 OpenMP: -Xcompiler -fopenmp

#include "cagraIndex.hpp"

using namespace cagra;

// ==========================================
// 1. 配置结构体
// ==========================================
struct BuildConfig {
    size_t data_size;
    int num_buckets;
    uint32_t degree;
};

struct SearchConfig {
    uint32_t itopk;
    uint32_t width;
    uint32_t min_iter;
    uint32_t max_iter;
};

// ==========================================
// 2. IO & GT Utils
// ==========================================
float* read_fvecs(const std::string& fname, size_t& d, size_t& n) {
    std::ifstream in(fname, std::ios::binary);
    if (!in.is_open()) { std::cerr << "Err " << fname << std::endl; exit(1); }
    int dim; in.read((char*)&dim, 4); d = dim;
    in.seekg(0, std::ios::end); size_t sz = in.tellg(); n = sz / (4 + d * 4);
    in.seekg(0, std::ios::beg);
    float* data = new float[n * d];
    for(size_t i=0; i<n; ++i) { in.read((char*)&dim,4); in.read((char*)(data+i*d), d*4); }
    return data;
}

inline float calc_l2_sq(const float* a, const float* b, int dim) {
    float r = 0; for(int i=0; i<dim; ++i) { float d = a[i]-b[i]; r+=d*d; } return r;
}

struct Neighbor { 
    int64_t id; 
    float dist; 
};

void compute_ground_truth_range(
    const float* dataset, const uint64_t* timestamps, size_t num_data,
    const float* queries, size_t num_queries,
    int dim, int target_k, 
    uint64_t start_ts, uint64_t end_ts,
    std::vector<std::vector<Neighbor>>& gt_vecs) 
{
    gt_vecs.resize(num_queries);
    #pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
        const float* q = queries + i * dim;
        std::vector<Neighbor> candidates;
        candidates.reserve(num_data / 10); 
        
        for (size_t j = 0; j < num_data; ++j) {
            if (timestamps[j] >= start_ts && timestamps[j] < end_ts) {
                float dist = calc_l2_sq(q, dataset + j * dim, dim);
                candidates.push_back({(int64_t)j, dist});
            }
        }
        
        size_t keep = std::min((size_t)target_k, candidates.size());
        if (candidates.size() > keep) {
            std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), 
                [](const Neighbor& a, const Neighbor& b){ return a.dist < b.dist; });
            candidates.resize(keep);
        } else {
             std::sort(candidates.begin(), candidates.end(), 
                [](const Neighbor& a, const Neighbor& b){ return a.dist < b.dist; });
        }
        gt_vecs[i] = std::move(candidates);
    }
}

// ==========================================
// 3. Test Logic
// ==========================================
void run_test(
    const float* full_data, size_t total_n, size_t dim,
    const std::vector<BuildConfig>& build_configs,
    const std::vector<SearchConfig>& search_configs
) {
    size_t num_queries = 1000;
    if (total_n <= num_queries) return;
    size_t num_base = total_n - num_queries;
    
    const float* base_data = full_data;
    const float* query_data = full_data + num_base * dim;

    size_t num_eval = 100; 
    int RECALL_K = 10; 
    int NUM_EXPERIMENTS = 5; // 执行 5 次独立实验取平均

    for (const auto& b_conf : build_configs) {
        size_t current_data_size = std::min(b_conf.data_size, num_base);
        std::cout << "\n=== BUILD: N=" << current_data_size << " Deg=" << b_conf.degree << " Bkts=" << b_conf.num_buckets << " ===" << std::endl;

        std::vector<uint64_t> timestamps(current_data_size);
        size_t chunk_size = (current_data_size + b_conf.num_buckets - 1) / b_conf.num_buckets;

        for (size_t i = 0; i < current_data_size; ++i) {
            timestamps[i] = i / chunk_size; 
            if (timestamps[i] >= b_conf.num_buckets) timestamps[i] = b_conf.num_buckets - 1;
        }

        CagraIndex index(dim, b_conf.degree);
        
        std::cout << "Inserting " << current_data_size << " vectors..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        index.insert(current_data_size, base_data, timestamps.data());
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Build time: " << std::chrono::duration<double>(t2 - t1).count() << "s" << std::endl;

        std::vector<float> range_ratios = {0.01f, 0.1f, 0.2f, 1.0f};
        
        // 预分配 buffer
        std::vector<int64_t> h_idxs(num_eval * 1024); // max itopk
        std::vector<float> h_dists(num_eval * 1024);

        for (const auto& s_conf : search_configs) {
            index.setQueryParams(s_conf.itopk, s_conf.width, s_conf.min_iter, s_conf.max_iter, 12);
            
            // 确保 buffer 够大
            if (h_idxs.size() < num_eval * s_conf.itopk) {
                h_idxs.resize(num_eval * s_conf.itopk);
                h_dists.resize(num_eval * s_conf.itopk);
            }

            for (float ratio : range_ratios) {
                double sum_recall = 0.0;
                double sum_latency = 0.0;

                // --- 5次独立实验循环 ---
                for (int run = 0; run < NUM_EXPERIMENTS; ++run) {
                    // 1. 重新生成随机范围
                    uint64_t total_range = b_conf.num_buckets;
                    uint64_t len = (uint64_t)(total_range * ratio);
                    if (len < 1) len = 1;
                    
                    uint64_t start = 0;
                    if (total_range > len) {
                        // 简单的随机生成，确保每次 run 可能不同
                        start = std::rand() % (total_range - len);
                    }
                    uint64_t end = start + len;

                    // 2. 重新计算 GT (耗时操作，OpenMP 必须开)
                    std::vector<std::vector<Neighbor>> gt_vecs;
                    compute_ground_truth_range(
                        base_data, timestamps.data(), current_data_size,
                        query_data, num_eval, dim, RECALL_K, 
                        start, end, gt_vecs
                    );

                    // 3. 执行查询 & 计时
                    auto t_q1 = std::chrono::high_resolution_clock::now();
                    index.query_range(
                        query_data, num_eval, RECALL_K, ratio,
                        start, end,
                        h_idxs.data(), h_dists.data()
                    );
                    auto t_q2 = std::chrono::high_resolution_clock::now();
                    double lat = std::chrono::duration<double, std::milli>(t_q2 - t_q1).count() / num_eval;
                    sum_latency += lat;

                    // 4. 计算本次实验的 Recall
                    double current_run_recall = 0;
                    int valid = 0;
                    for(size_t i=0; i<num_eval; ++i) {
                        if (gt_vecs[i].empty()) continue;
                        
                        // 构建 Set 加速查找
                        std::set<int64_t> gt_set;
                        for (const auto& n : gt_vecs[i]) gt_set.insert(n.id);

                        int hit = 0;
                        for(int k=0; k<RECALL_K; ++k) {
                            if (gt_set.count(h_idxs[i*RECALL_K + k])) hit++;
                        }
                        current_run_recall += (double)hit / std::min((int)gt_vecs[i].size(), RECALL_K);
                        valid++;
                    }
                    sum_recall += (valid > 0) ? (current_run_recall / valid) : 1.0;
                }
                
                // --- 计算 5 次平均值 ---
                double avg_recall = sum_recall / NUM_EXPERIMENTS;
                double avg_latency = sum_latency / NUM_EXPERIMENTS;

                std::cout << "  Search [" << s_conf.itopk << ", " << s_conf.max_iter << "] "
                          << "Range " << std::setw(3) << (int)(ratio*100) << "% "
                          << "R@" << RECALL_K << ": " << std::fixed << std::setprecision(4) << avg_recall
                          << " Lat: " << std::setprecision(3) << avg_latency << " ms" 
                          << " (Avg of " << NUM_EXPERIMENTS << " runs)" << std::endl;
            }
        }
    }
}

int main() {
    std::srand(42);
    size_t dim, n;
    float* data = read_fvecs("../data/sift.fvecs", dim, n);
    
    // 构建配置
    std::vector<BuildConfig> b_confs = {
        {1000000, 100, 32}
    };

    // 搜索配置
    std::vector<SearchConfig> s_confs = {
        {128, 4, 0, 50},
        {128, 4, 0, 100},
        {128, 4, 0, 150},
        {128, 4, 0, 200},
        {256, 4, 0, 50},
        {256, 4, 0, 100},
        {256, 4, 0, 150},
        {256, 4, 0, 200},
        {512, 4, 0, 50},
        {512, 4, 0, 100},
        {512, 4, 0, 150},
        {512, 4, 0, 200},
    };

    run_test(data, n, dim, b_confs, s_confs);
    delete[] data;
    return 0;
}