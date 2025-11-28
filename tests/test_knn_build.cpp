#include "cagra.cuh"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <set>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main() {
    // 1. 配置
    const size_t N = 10000;    
    const uint32_t DIM = 1024;  
    const uint32_t K = 10;     
    
    std::cout << "==================================================" << std::endl;
    std::cout << "Running KNN Build Test (Output to CPU)" << std::endl;
    std::cout << "Data Size: " << N << ", Dim: " << DIM << ", K: " << K << std::endl;

    // 2. 生成有序数据 (CPU)
    std::vector<float> h_dataset(N * DIM);
    for (size_t i = 0; i < N; ++i) {
        float val = static_cast<float>(i) * 0.001f; 
        for (uint32_t d = 0; d < DIM; ++d) {
            h_dataset[i * DIM + d] = val;
        }
    }

    // 3. 准备 GPU 数据集
    float* d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, N * DIM * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_dataset, h_dataset.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice));

    // 4. 准备结果缓冲区 (CPU !!!)
    // 直接在 Host 上分配内存来接收结果
    std::vector<uint32_t> h_knn_graph(N * K);

    // 5. 调用函数
    std::cout << ">> Calling generate_knn_graph..." << std::endl;
    
    // 传入 d_dataset (GPU) 和 h_knn_graph.data() (CPU)
    cagra::generate_knn_graph(d_dataset, N, DIM, K, h_knn_graph.data());
    
    std::cout << ">> Graph generation finished." << std::endl;

    // 6. 验证结果
    int correct_count = 0;
    int total_neighbors = N * K;
    int self_loop_errors = 0;

    for (size_t i = 0; i < N; ++i) {
        std::vector<uint32_t> actual_neighbors;
        for (uint32_t k = 0; k < K; ++k) {
            actual_neighbors.push_back(h_knn_graph[i * K + k]);
        }

        // 检查 Self-Loop
        for (uint32_t neighbor : actual_neighbors) {
            if (neighbor == i) self_loop_errors++;
        }

        // 检查 Ground Truth
        // 标准答案：距离 i 最近的 K 个点
        std::set<uint32_t> ideal_neighbors;
        int delta = 1;
        while (ideal_neighbors.size() < K) {
            if (static_cast<int>(i) - delta >= 0) ideal_neighbors.insert(i - delta);
            if (ideal_neighbors.size() >= K) break;
            if (i + delta < N) ideal_neighbors.insert(i + delta);
            delta++;
        }

        for (uint32_t neighbor : actual_neighbors) {
            if (ideal_neighbors.count(neighbor)) {
                correct_count++;
            }
        }

        if (i < 5) {
            std::cout << "Query " << i << " Neighbors: ";
            for (auto n : actual_neighbors) std::cout << n << " ";
            std::cout << "| Ideal: ";
            for (auto n : ideal_neighbors) std::cout << n << " ";
            std::cout << std::endl;
        }
    }

    double recall = 100.0 * correct_count / total_neighbors;

    std::cout << "==================================================" << std::endl;
    std::cout << "Test Report:" << std::endl;
    std::cout << "Total Queries: " << N << std::endl;
    std::cout << "Total Neighbors Checked: " << total_neighbors << std::endl;
    std::cout << "Self-Loop Errors: " << self_loop_errors << " (Should be 0)" << std::endl;
    std::cout << "Recall: " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    
    CHECK_CUDA(cudaFree(d_dataset));
    // h_knn_graph 不需要 cudaFree，它是 vector

    if (self_loop_errors == 0 && recall > 99.0) {
        std::cout << "PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
}