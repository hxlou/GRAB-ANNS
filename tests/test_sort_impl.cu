// 引入必要的头文件
#include "search.cuh" 
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
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
// 测试 Kernel wrapper
// =============================================================================
template <uint32_t CAPACITY>
__global__ void test_load_sort_store_kernel(float* d_dists, uint32_t* d_indices) {
    // 1. 申请 Shared Memory
    // 我们需要两个数组：dists 和 indices
    extern __shared__ uint8_t smem[];
    float* smem_dists = (float*)smem;
    uint32_t* smem_indices = (uint32_t*)(smem_dists + CAPACITY);

    // 2. 将 Global Memory 数据加载到 Shared Memory (模拟搜索前的状态)
    // 使用 Block 内所有线程协作加载
    for (int i = threadIdx.x; i < CAPACITY; i += blockDim.x) {
        smem_dists[i] = d_dists[i];
        smem_indices[i] = d_indices[i];
    }
    __syncthreads();

    // 3. 调用待测函数 (只让 Warp 0 跑)
    if (threadIdx.x < 32) {
        constexpr int N = CAPACITY / 32;
        // 调用 cagra::device 命名空间下的函数
        cagra::device::load_sort_store<N>(smem_dists, smem_indices, CAPACITY);
    }
    __syncthreads();

    // 4. 将结果写回 Global Memory
    for (int i = threadIdx.x; i < CAPACITY; i += blockDim.x) {
        d_dists[i] = smem_dists[i];
        d_indices[i] = smem_indices[i];
    }
}

// =============================================================================
// Host 端测试逻辑
// =============================================================================
template <uint32_t CAPACITY>
void run_test(const char* name) {
    std::cout << ">> Running Test: " << name << " (Cap=" << CAPACITY << ")" << std::endl;

    // 1. 准备随机数据
    std::vector<float> h_dists(CAPACITY);
    std::vector<uint32_t> h_indices(CAPACITY);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1000.0f);

    // Ground Truth 用于验证
    std::vector<std::pair<float, uint32_t>> gt(CAPACITY);

    std::cout << "   Input (First 8): ";
    for (int i = 0; i < CAPACITY; ++i) {
        h_dists[i] = dis(gen);
        h_indices[i] = i; // 原始索引作为 ID，方便查错
        gt[i] = {h_dists[i], h_indices[i]};
        
        if (i < 8) std::cout << std::fixed << std::setprecision(1) << h_dists[i] << " ";
    }
    std::cout << "..." << std::endl;

    // 计算真值
    std::sort(gt.begin(), gt.end());

    // 2. GPU 内存
    float* d_dists;
    uint32_t* d_indices;
    CHECK_CUDA(cudaMalloc(&d_dists, CAPACITY * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, CAPACITY * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(d_dists, h_dists.data(), CAPACITY * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), CAPACITY * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 3. 启动 Kernel
    // Shared Mem 大小 = Capacity * (sizeof(float) + sizeof(uint32))
    size_t smem_size = CAPACITY * (sizeof(float) + sizeof(uint32_t));
    
    // Block Size 设为 256 (模拟真实场景)
    test_load_sort_store_kernel<CAPACITY><<<1, 256, smem_size>>>(d_dists, d_indices);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 结果验证
    std::vector<float> h_dists_out(CAPACITY);
    std::vector<uint32_t> h_indices_out(CAPACITY);
    CHECK_CUDA(cudaMemcpy(h_dists_out.data(), d_dists, CAPACITY * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_indices_out.data(), d_indices, CAPACITY * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::cout << "   Output(First 8): ";
    for(int i=0; i<8; ++i) std::cout << h_dists_out[i] << " ";
    std::cout << "..." << std::endl;

    bool passed = true;
    for (int i = 0; i < CAPACITY; ++i) {
        // 允许微小的浮点误差，或者相同 key 的顺序差异
        float diff = std::abs(h_dists_out[i] - gt[i].first);
        if (diff > 1e-4) {
            passed = false;
            std::cout << "   [FAIL] Mismatch at index " << i << std::endl;
            std::cout << "          Expected: " << gt[i].first << " (" << gt[i].second << ")" << std::endl;
            std::cout << "          Got:      " << h_dists_out[i] << " (" << h_indices_out[i] << ")" << std::endl;
            break; // 只要错一个就停
        }
    }

    if (passed) {
        std::cout << "   [PASS] Sort is correct." << std::endl;
    } else {
        std::cout << "   [FAIL] Test Failed!" << std::endl;
        exit(1);
    }

    CHECK_CUDA(cudaFree(d_dists));
    CHECK_CUDA(cudaFree(d_indices));
}

int main() {
    // 测试 TopK=64 (N=2)
    run_test<64>("Capacity=64 (N=2)");
    std::cout << "------------------------------------------" << std::endl;

    // 测试 TopK=128 (N=4) - 你的默认配置
    run_test<128>("Capacity=128 (N=4)");
    std::cout << "------------------------------------------" << std::endl;

    // 测试 TopK=256 (N=8)
    run_test<256>("Capacity=256 (N=8)");
    std::cout << "------------------------------------------" << std::endl;

    // 测试 TopK=512 (N=16)
    run_test<512>("Capacity=512 (N=16)");

    return 0;
}