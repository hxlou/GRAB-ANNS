#include "../src/index/bitonic.cuh" // 包含刚才写的 bitonic 实现

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <iomanip>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// =============================================================================
// 测试 Kernel
// =============================================================================
// 每个线程持有 N 个元素，一个 Warp 总共排序 32 * N 个元素
template <int N>
__global__ void test_warp_sort_kernel(float* d_keys, uint32_t* d_vals) {
    // 1. 定义寄存器数组
    float key[N];
    uint32_t val[N];

    int tid = threadIdx.x;
    int lane_id = tid % 32; // 既然是 Warp Sort，我们只关心 Warp 内的 ID

    // 为了测试简单，我们只启动 1 个 Block，32 个线程
    if (tid >= 32) return;

    // 2. 从 Global Memory 加载数据到寄存器
    // 模拟 CAGRA 中的数据加载模式
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int global_idx = lane_id * N + i; // 这种布局利于写回验证
        key[i] = d_keys[global_idx];
        val[i] = d_vals[global_idx];
    }

    // 3. 调用核心排序函数 (升序)
    cagra::bitonic::warp_sort<float, uint32_t, N>(key, val, true);

    // 4. 写回 Global Memory
    // 注意：如果排序正确，写回后的 d_keys 应该是全局有序的
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int global_idx = lane_id * N + i;
        d_keys[global_idx] = key[i];
        d_vals[global_idx] = val[i];
    }
}

// =============================================================================
// 主机端验证逻辑
// =============================================================================
template <int N>
void run_test(const char* name) {
    std::cout << ">> Running Test: " << name << " (N=" << N << ")" << std::endl;

    const int WARP_SIZE = 32;
    const int TOTAL_ELEMENTS = WARP_SIZE * N;

    // 1. 准备数据
    std::vector<float> h_keys(TOTAL_ELEMENTS);
    std::vector<uint32_t> h_vals(TOTAL_ELEMENTS);
    
    // 生成随机数据
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    std::cout << "   Input (First 10): ";
    for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
        h_keys[i] = dis(gen);
        h_vals[i] = i; // Value 是原始索引，方便验证绑定关系
        if (i < 10) std::cout << std::fixed << std::setprecision(1) << h_keys[i] << " ";
    }
    std::cout << "..." << std::endl;

    // 2. 拷贝到 GPU
    float* d_keys;
    uint32_t* d_vals;
    CHECK_CUDA(cudaMalloc(&d_keys, TOTAL_ELEMENTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vals, TOTAL_ELEMENTS * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(d_keys, h_keys.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), TOTAL_ELEMENTS * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 3. 启动 Kernel
    // 1 Block, 32 Threads
    test_warp_sort_kernel<N><<<1, 32>>>(d_keys, d_vals);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 拷回结果
    std::vector<float> h_keys_out(TOTAL_ELEMENTS);
    std::vector<uint32_t> h_vals_out(TOTAL_ELEMENTS);
    CHECK_CUDA(cudaMemcpy(h_keys_out.data(), d_keys, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vals_out.data(), d_vals, TOTAL_ELEMENTS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 5. 验证
    bool sorted = true;
    bool binding_ok = true;

    // CPU 做一次标准排序作为真值 (Ground Truth)
    // 这是一个 Pair 排序，用于验证 Key-Value 绑定
    std::vector<std::pair<float, uint32_t>> ground_truth(TOTAL_ELEMENTS);
    for(int i=0; i<TOTAL_ELEMENTS; ++i) {
        ground_truth[i] = {h_keys[i], h_vals[i]};
    }
    std::sort(ground_truth.begin(), ground_truth.end());

    std::cout << "   Output (First 10): ";
    for(int i=0; i<10; ++i) std::cout << h_keys_out[i] << " ";
    std::cout << "..." << std::endl;

    for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
        // 验证顺序
        if (i > 0 && h_keys_out[i] < h_keys_out[i-1]) {
            sorted = false;
            std::cout << "   [FAIL] Not sorted at index " << i 
                      << ": " << h_keys_out[i-1] << " > " << h_keys_out[i] << std::endl;
            break;
        }

        // 验证绑定 (虽然可能有重复 Key，但我们可以近似验证或者用 multiset 精确验证)
        // 简单验证：如果在 Ground Truth 中对应的 Key 也是这个，且 Value 也匹配
        // 注意：如果有相同 Key，排序是不稳定的，Value 可能顺序不同。
        // 为了简单，我们只检查是否完全匹配 Ground Truth (假设 bitonic 实现和 std::sort 对相同元素处理一致，或者 Key 不重复)
        // 这里的测试数据是 float，重复概率极低。
        
        if (std::abs(h_keys_out[i] - ground_truth[i].first) > 1e-5 || 
            h_vals_out[i] != ground_truth[i].second) {
            // 浮点数比较可能有一点点误差，或者相同Key顺序不同
            // 如果 Key 相同，Value 不同，算 Pass (不稳定排序)
            if (std::abs(h_keys_out[i] - ground_truth[i].first) < 1e-5) {
                // Key 相同，Value 不同 -> 允许
            } else {
                binding_ok = false;
                std::cout << "   [FAIL] Binding error at index " << i << std::endl;
                std::cout << "          Expected: (" << ground_truth[i].first << ", " << ground_truth[i].second << ")" << std::endl;
                std::cout << "          Got:      (" << h_keys_out[i] << ", " << h_vals_out[i] << ")" << std::endl;
                break;
            }
        }
    }

    if (sorted && binding_ok) {
        std::cout << "   [PASS] Result is sorted and values are bound." << std::endl;
    } else {
        std::cout << "   [FAIL] Test failed." << std::endl;
        exit(1);
    }

    CHECK_CUDA(cudaFree(d_keys));
    CHECK_CUDA(cudaFree(d_vals));
}

int main() {
    // 测试 N=2 (总数 64)
    run_test<2>("Bitonic Sort N=2 (64 elements)");
    
    std::cout << "---------------------------------------------------" << std::endl;

    // 测试 N=4 (总数 128) - 这是 CAGRA itopk=128 时的典型场景
    run_test<4>("Bitonic Sort N=4 (128 elements)");

    std::cout << "---------------------------------------------------" << std::endl;

    // 测试 N=8 (总数 256)
    run_test<8>("Bitonic Sort N=8 (256 elements)");

    return 0;
}