#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> // for std::generate

// 引入 RAFT 和 RMM 必要的头文件
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

// 引入你的头文件
#include "index/raft_help.cuh"

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
    // ---------------------------------------------------------
    // 【修复 1】必须初始化 RMM，否则头文件里的 device_uvector 会报错
    // ---------------------------------------------------------


    // ---------------------------------------------------------
    // 【修复 2】创建 RAFT 句柄
    // ---------------------------------------------------------
    raft::device_resources res;

    uint32_t buckets = 100;
    uint32_t dim = 128;
    uint32_t nums_per_bucket = 1000;
    uint32_t degree = 32;

    std::cout << "Initializing host data..." << std::endl;
    std::vector<float> data(buckets * nums_per_bucket * dim);
    std::vector<size_t> bucket_sizes(buckets, nums_per_bucket);
    std::vector<uint32_t> graph(buckets * nums_per_bucket * degree);

    // 填充随机数据 (稍微优化一下写法)
    for (auto& v : data) {
        v = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float* d_data;
    uint32_t* d_graph;
    CHECK_CUDA(cudaMalloc(&d_data, data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_graph, graph.size() * sizeof(uint32_t)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    // 清零图数据 (其实不必要，因为会被覆盖，但好习惯)
    CHECK_CUDA(cudaMemset(d_graph, 0, graph.size() * sizeof(uint32_t)));

    std::cout << "Running RAFT batch KNN..." << std::endl;

    // ---------------------------------------------------------
    // 【修复 3】调用时传入 res 句柄，且确保函数名正确
    // 注意：请确保你在 raft_help.cuh 里定义的函数名是 
    // build_batch_knn_graphs 还是 build_knn_graphs
    // 这里假设你改成了 build_batch_knn_graphs
    // ---------------------------------------------------------
    build_batch_knn_graphs( // <--- 请检查你的头文件里函数叫什么名字，对应修改这里
        res,          // <--- 加上这个参数
        d_data,
        dim,
        bucket_sizes,
        degree,
        d_graph
    );

    // 验证结果
    std::cout << "Copying results back..." << std::endl;
    CHECK_CUDA(cudaMemcpy(graph.data(), d_graph, graph.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // 简单验证前3个桶
    for (size_t b = 0; b < 3; ++b) {
        size_t offset = b * nums_per_bucket * degree;
        std::cout << "--- Bucket " << b << " Sample ---" << std::endl;
        for (size_t i = 0; i < 5; ++i) {
            std::cout << "Node " << i << ": [ ";
            for (size_t j = 0; j < degree; ++j) {
                // 验证索引是否越界 (应该是 0 到 nums_per_bucket-1)
                uint32_t neighbor_idx = graph[offset + i * degree + j];
                std::cout << neighbor_idx << " ";
                if (neighbor_idx >= nums_per_bucket) {
                    std::cerr << "Error: Index out of bound!" << std::endl;
                }
            }
            std::cout << "]" << std::endl;
        }
    }

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_graph));
    
    // res 会自动析构，不需要手动释放
    // cuda_mr 也是栈上对象，自动析构
    
    return 0;
}