#include "timeStampIndex.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <set>
#include <cassert>
#include <omp.h>

// 使用 CPU FAISS 生成 GT (避免显存 OOM)
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
        cudaDeviceSynchronize(); // 确保 GPU 任务完成
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main() {
    // ---------------------------------------------------------
    // 1. 配置参数
    // ---------------------------------------------------------
    const size_t N = 5000000;        // 500万数据
    const uint32_t DIM = 1024;       // 1024 维
    const size_t NUM_QUERIES = 100;  // 100 个查询
    const int TOPK = 20;             // Top-20
    const int BUCKET_SIZE = 50000;   // 每个桶 5w 数据
    
    // 查询配置
    const int PROBE_BUCKETS = 50;    // 探测 50 个桶 (总共 100 个)

    std::cout << "==========================================================" << std::endl;
    std::cout << "Large Scale Test: 5M Vectors, 1024 Dim" << std::endl;
    std::cout << "Memory Required: ~20GB for raw data + Overhead" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ---------------------------------------------------------
    // 2. 生成随机数据 (Host RAM)
    // ---------------------------------------------------------
    std::cout << ">> [Init] Allocating " << (N * DIM * sizeof(float) / 1024.0 / 1024.0 / 1024.0) << " GB RAM..." << std::endl;
    
    // 使用 vector 可能会触发 alloc 异常，建议 try-catch
    std::vector<float> host_data;
    try {
        host_data.resize(N * DIM);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate memory for dataset: " << e.what() << std::endl;
        return 1;
    }

    std::cout << ">> [Init] Generating Random Data (OpenMP)..." << std::endl;
    #pragma omp parallel 
    {
        std::mt19937 rng(42 + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(-1.0, 1.0);
        
        #pragma omp for
        for (size_t i = 0; i < N * DIM; ++i) {
            host_data[i] = dis(rng);
        }
    }

    // ---------------------------------------------------------
    // 3. 生成时间戳 & 准备 Query
    // ---------------------------------------------------------
    std::vector<uint64_t> timestamps(N);
    size_t num_buckets = (N + BUCKET_SIZE - 1) / BUCKET_SIZE;
    
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        timestamps[i] = i / BUCKET_SIZE;
    }
    std::cout << "   Total Buckets: " << num_buckets << " (Expected 100)" << std::endl;

    // 随机抽取查询向量
    std::vector<float> queries(NUM_QUERIES * DIM);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<size_t> dist(0, N - 1);
    
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        size_t idx = dist(rng);
        std::copy(host_data.begin() + idx * DIM, 
                  host_data.begin() + (idx + 1) * DIM, 
                  queries.begin() + i * DIM);
    }

    // ---------------------------------------------------------
    // 4. 生成 Ground Truth (CPU FAISS)
    // ---------------------------------------------------------
    std::cout << ">> [GT] Generating Ground Truth using CPU FAISS..." << std::endl;
    std::vector<int64_t> gt_indices(NUM_QUERIES * TOPK);
    std::vector<float> gt_dists(NUM_QUERIES * TOPK);

    Timer timer;
    {
        // 使用 CPU IndexFlatL2，避免 GPU 显存爆炸
        faiss::IndexFlatL2 cpu_index(DIM);
        cpu_index.add(N, host_data.data()); // 零拷贝添加
        cpu_index.search(NUM_QUERIES, queries.data(), TOPK, gt_dists.data(), gt_indices.data());
    }
    std::cout << "   GT Generation Time: " << timer.elapsed_ms() / 1000.0 << " s" << std::endl;

    // ---------------------------------------------------------
    // 5. [统计] GT 的桶分布情况
    // ---------------------------------------------------------
    std::cout << "\n>> [Analysis] Analyzing GT Bucket Distribution..." << std::endl;
    
    double total_buckets_hit = 0.0;
    
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        std::unordered_set<uint64_t> unique_buckets;
        for (int j = 0; j < TOPK; ++j) {
            int64_t gid = gt_indices[i * TOPK + j];
            if (gid >= 0 && gid < (int64_t)N) {
                unique_buckets.insert(timestamps[gid]);
            }
        }
        total_buckets_hit += (double)unique_buckets.size();
    }

    double avg_buckets = total_buckets_hit / NUM_QUERIES;
    double avg_pct = (avg_buckets / num_buckets) * 100.0;

    std::cout << "   Avg Buckets Hit per Query: " << std::fixed << std::setprecision(2) 
              << avg_buckets << " / " << num_buckets << std::endl;
    std::cout << "   Avg Distribution: " << avg_pct << "% of total buckets." << std::endl;
    std::cout << "   Probe Buckets Setting: " << PROBE_BUCKETS << std::endl;
    
    if (PROBE_BUCKETS < avg_buckets) {
        std::cout << "   [WARNING] Probe buckets < Avg GT spread. Recall will be limited!" << std::endl;
    } else {
        std::cout << "   [OK] Probe coverage is sufficient." << std::endl;
    }
    std::cout << "----------------------------------------------------------" << std::endl;

    // ---------------------------------------------------------
    // 6. TimeStampIndex 测试
    // ---------------------------------------------------------
    std::cout << "\n>> [TimeStampIndex] Start Testing..." << std::endl;
    
    // 初始化：cluster_ratio=1000 => 每个桶5w数据生成50个虚拟点
    // 总共100个桶 => 5000个虚拟点，这对于顶层图来说规模很小，速度很快
    timestamp::TimeStampIndex ts_index(DIM, 32, 1000);

    // Step A: Insert (Host Copy)
    std::cout << ">> Step 1: Inserting 5M vectors..." << std::endl;
    timer.reset();
    ts_index.insert(host_data.data(), timestamps.data(), N);
    double t_insert = timer.elapsed_ms();
    std::cout << "   Insert Time: " << t_insert / 1000.0 << " s" << std::endl;

    // 释放原始 host_data 以节省内存 (如果内存紧张的话)
    // host_data.clear(); host_data.shrink_to_fit(); 
    // 注意：如果释放了，上面 query 就不能用了，需要提前拷贝 query。这里暂时保留。

    // Step B: Build Virtual Index (GPU KMeans + CAGRA Build)
    std::cout << ">> Step 2: Building Virtual Layer..." << std::endl;
    timer.reset();
    ts_index.build_virtual_index();
    double t_build = timer.elapsed_ms();
    std::cout << "   Build Time: " << t_build / 1000.0 << " s" << std::endl;

    // Step C: Query
    std::cout << ">> Step 3: Querying (Probe=" << PROBE_BUCKETS << ")..." << std::endl;
    std::vector<int64_t> out_indices(NUM_QUERIES * TOPK);
    std::vector<float> out_dists(NUM_QUERIES * TOPK);

    // 预热
    ts_index.query(queries.data(), 1, out_indices.data(), out_dists.data(), PROBE_BUCKETS);

    timer.reset();
    // 串行调用 query 接口 (内部有 OpenMP 并行搜桶)
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        ts_index.query(queries.data() + i * DIM, 
                       TOPK, 
                       out_indices.data() + i * TOPK, 
                       out_dists.data() + i * TOPK, 
                       PROBE_BUCKETS);
    }
    double t_search = timer.elapsed_ms();

    // ---------------------------------------------------------
    // 7. 最终统计
    // ---------------------------------------------------------
    int correct_cnt = 0;
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < TOPK; ++j) gt_set.insert(gt_indices[i * TOPK + j]);
        
        for (int j = 0; j < TOPK; ++j) {
            if (gt_set.count(out_indices[i * TOPK + j])) correct_cnt++;
        }
    }
    double recall = 100.0 * correct_cnt / (NUM_QUERIES * TOPK);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "   Total Vectors: " << ts_index.size() << std::endl;
    std::cout << "   Search Time:   " << t_search << " ms" << std::endl;
    std::cout << "   QPS:           " << std::fixed << std::setprecision(2) << (NUM_QUERIES * 1000.0 / t_search) << std::endl;
    std::cout << "   Latency:       " << (t_search / NUM_QUERIES) << " ms/query" << std::endl;
    std::cout << "   Recall@" << TOPK << ":      " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}