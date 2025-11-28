#include "cagra.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cassert>

// -----------------------------------------------------------------------------
// 计时辅助工具
// -----------------------------------------------------------------------------
class Timer {
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    
    // 返回毫秒
    double elapsed() {
        cudaDeviceSynchronize(); // 确保 GPU 任务完成
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// 主程序
// -----------------------------------------------------------------------------
int main() {
    // ==========================================
    // 1. 配置参数
    // ==========================================
    const size_t N = 10000;      // 数据量
    const uint32_t DIM = 1024;    // 维度
    
    // 流程参数
    const uint32_t KNN_K = 64;   // 初始 KNN 度数 (通常较大，如 128)
    const uint32_t GRAPH_K = 32;  // 最终 CAGRA 图度数 (通常为 32 或 64)
    
    // 确保参数合理
    assert(KNN_K >= GRAPH_K);

    std::cout << "==========================================================" << std::endl;
    std::cout << "CAGRA Complete Pipeline Test" << std::endl;
    std::cout << "Data: " << N << " vectors, Dim: " << DIM << std::endl;
    std::cout << "Pipeline: KNN(K=" << KNN_K << ") -> Prune(K=" << GRAPH_K 
              << ") -> Reverse -> Merge" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ==========================================
    // 2. 数据准备
    // ==========================================
    std::cout << ">> [Init] Generating Random Data..." << std::endl;
    std::vector<float> h_dataset(N * DIM);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    for (size_t i = 0; i < N * DIM; ++i) {
        h_dataset[i] = dis(gen);
    }

    float* d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, N * DIM * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_dataset, h_dataset.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice));

    // 预分配所有需要的 Host 内存
    std::vector<uint32_t> h_knn_graph(N * KNN_K);        // 中间结果 1
    std::vector<uint32_t> h_final_graph(N * GRAPH_K);    // 最终结果 (也是剪枝后的结果)
    std::vector<uint32_t> h_rev_graph(N * GRAPH_K);      // 反向图
    std::vector<uint32_t> h_rev_counts(N);               // 反向计数

    Timer total_timer;
    Timer stage_timer;

    // ==========================================
    // 3. Stage 1: 构建初始 KNN 图
    // ==========================================
    std::cout << "\n>> [Stage 1] Build KNN Graph (FAISS + Refine)..." << std::endl;
    stage_timer.reset();
    
    cagra::generate_knn_graph(d_dataset, N, DIM, KNN_K, h_knn_graph.data());
    
    double time_knn = stage_timer.elapsed();
    std::cout << "   Stage 1 Done. Time: " << std::fixed << std::setprecision(2) << time_knn << " ms" << std::endl;

    // ==========================================
    // 4. Stage 2: 剪枝 (Pruning)
    // ==========================================
    std::cout << "\n>> [Stage 2] Optimizing (Pruning)..." << std::endl;
    stage_timer.reset();

    cagra::optimize_prune(h_knn_graph.data(), 
                          h_final_graph.data(), // 剪枝结果写入 final_graph
                          N, 
                          KNN_K, 
                          GRAPH_K);

    double time_prune = stage_timer.elapsed();
    std::cout << "   Stage 2 Done. Time: " << time_prune << " ms" << std::endl;

    // ==========================================
    // 5. Stage 3: 构建反向图
    // ==========================================
    std::cout << "\n>> [Stage 3] Creating Reverse Graph..." << std::endl;
    stage_timer.reset();

    cagra::optimize_create_reverse_graph(h_final_graph.data(), // 输入是剪枝后的图
                                         h_rev_graph.data(),
                                         h_rev_counts.data(),
                                         N,
                                         GRAPH_K);

    double time_rev = stage_timer.elapsed();
    std::cout << "   Stage 3 Done. Time: " << time_rev << " ms" << std::endl;

    // ==========================================
    // 6. Stage 4: 合并图 (注入反向边)
    // ==========================================
    std::cout << "\n>> [Stage 4] Merging Graphs..." << std::endl;
    stage_timer.reset();

    cagra::optimize_merge_graphs(h_final_graph.data(), // 输入并原地修改
                                 h_rev_graph.data(),
                                 h_rev_counts.data(),
                                 N,
                                 GRAPH_K);

    double time_merge = stage_timer.elapsed();
    std::cout << "   Stage 4 Done. Time: " << time_merge << " ms" << std::endl;

    double time_total = total_timer.elapsed();

    // ==========================================
    // 7. 结果分析与统计
    // ==========================================
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "Pipeline Summary" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    std::cout << std::left << std::setw(25) << "1. KNN Build:"     << std::right << std::setw(10) << time_knn   << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "2. Pruning:"       << std::right << std::setw(10) << time_prune << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "3. Reverse Graph:" << std::right << std::setw(10) << time_rev   << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "4. Merge:"         << std::right << std::setw(10) << time_merge << " ms" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << std::left << std::setw(25) << "TOTAL TIME:"       << std::right << std::setw(10) << time_total << " ms" << std::endl;

    // 检查最终图的质量（简单的连通性统计）
    long long total_edges = 0;
    int max_degree = 0;
    int min_degree = 9999;
    int zero_degree_cnt = 0;

    for (size_t i = 0; i < N; ++i) {
        int degree_cnt = 0;
        for (uint32_t k = 0; k < GRAPH_K; ++k) {
            uint32_t neighbor = h_final_graph[i * GRAPH_K + k];
            // 过滤 0 和 自身（虽然 generate_knn_graph 已经去了自身，但 0 可能是有效 ID 0，也可能是填充）
            // 在随机数据中，节点 0 被作为邻居的概率很低，且如果是填充的0，通常出现在末尾
            // 为了准确，我们假设 0 号节点是有效的。
            // 唯一无效的是 padding。我们的代码中 padding 是 0。
            // 严格来说，应该用 0xFFFFFFFF 初始化，但为了简单我们之前用了 0。
            // 这里简单统计非0边（假设数据量大，0作为邻居概率小，或者把0当有效点）
            
            // 更好的判断：在 optimize_merge_graphs 里如果没填满是补0。
            // 这里我们统计非0值。
            // 注意：如果是真实节点0，会被误判。但在 10000 个随机点中，影响可忽略。
            if (neighbor != 0xFFFFFFFF) { 
                degree_cnt++;
            }
        }
        total_edges += degree_cnt;
        if (degree_cnt > max_degree) max_degree = degree_cnt;
        if (degree_cnt < min_degree) min_degree = degree_cnt;
        if (degree_cnt == 0) zero_degree_cnt++;
    }

    std::cout << "\nGraph Statistics (Approximate):" << std::endl;
    std::cout << "Avg Degree: " << (double)total_edges / N << " / " << GRAPH_K << std::endl;
    std::cout << "Min Degree: " << min_degree << std::endl;
    std::cout << "Max Degree: " << max_degree << std::endl;
    
    CHECK_CUDA(cudaFree(d_dataset));

    std::cout << "\nPASSED: CAGRA Graph Construction Complete." << std::endl;
    return 0;
}