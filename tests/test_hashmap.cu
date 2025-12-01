#include "../src/index/hashmap.cuh" // 假设你的哈希表头文件在这里
#include <iostream>
#include <vector>
#include <cassert>
#include <random>

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
// 我们启动 1 个 Block，使用 Shared Memory 模拟 CAGRA 的真实用法
__global__ void test_hashmap_kernel(
    uint32_t hash_bitlen, 
    uint32_t num_insertions, 
    const uint32_t* input_keys, 
    uint32_t* output_results, // 0: Success, 1: Duplicate, 2: Failed/Full
    uint32_t* debug_table     // 用于回传 Hashmap 内容供检查
) {
    // 1. 动态申请 Shared Memory
    extern __shared__ uint32_t smem_table[];

    // 2. 初始化 Hashmap
    // init 函数设计为利用当前 Block 所有线程并行初始化
    cagra::hashmap::init(smem_table, hash_bitlen);
    __syncthreads(); // 必须同步，等待初始化完成

    // 3. 并发插入测试
    // grid-stride loop (虽然这里只有1个block)
    for (int i = threadIdx.x; i < num_insertions; i += blockDim.x) {
        uint32_t key = input_keys[i];
        
        // 调用 insert
        bool success = cagra::hashmap::insert(smem_table, hash_bitlen, key);

        // 记录结果
        // 这里只是为了验证 insert 的返回值是否正确
        // 实际上我们会通过原子计数器统计总成功数
        if (success) {
            output_results[i] = 1; // Inserted (New)
        } else {
            output_results[i] = 0; // Duplicated or Full
        }
    }
    __syncthreads();

    // 4. 将 Shared Memory 的内容拷回 Global Memory 供 Host 检查 (可选)
    uint32_t size = cagra::hashmap::compute_size(hash_bitlen);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        debug_table[i] = smem_table[i];
    }
}

// =============================================================================
// Host 端测试逻辑
// =============================================================================
void run_test(uint32_t bitlen, uint32_t num_unique_items, float duplicate_ratio) {
    uint32_t capacity = 1u << bitlen;
    std::cout << ">> Testing Hashmap (Bitlen=" << bitlen << ", Cap=" << capacity << ")" << std::endl;

    // 1. 生成测试数据
    // 我们生成 num_unique_items 个唯一 Key
    // 然后混入一些重复 Key
    std::vector<uint32_t> keys;
    std::vector<uint32_t> expected_results; // 1=ShouldSucceed, 0=ShouldFail

    // 生成唯一 Key (从 1 开始，因为 0xFFFFFFFF 是 INVALID)
    for (uint32_t i = 0; i < num_unique_items; ++i) {
        keys.push_back(i + 1); 
        expected_results.push_back(1); // 第一次插入应该成功
    }

    // 生成重复 Key
    uint32_t num_duplicates = (uint32_t)(num_unique_items * duplicate_ratio);
    for (uint32_t i = 0; i < num_duplicates; ++i) {
        // 重复插入前 num_duplicates 个 Key
        keys.push_back(i + 1);
        expected_results.push_back(0); // 第二次插入应该失败
    }

    uint32_t total_ops = keys.size();
    std::cout << "   Total Ops: " << total_ops 
              << " (Unique: " << num_unique_items << ", Dups: " << num_duplicates << ")" 
              << " Load Factor: " << (float)num_unique_items/capacity << std::endl;

    if (num_unique_items > capacity) {
        std::cerr << "   [WARNING] Items > Capacity. Some inserts MUST fail (Full)." << std::endl;
    }

    // 2. 准备 GPU 内存
    uint32_t* d_keys;
    uint32_t* d_results;
    uint32_t* d_debug_table;

    CHECK_CUDA(cudaMalloc(&d_keys, total_ops * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_results, total_ops * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_debug_table, capacity * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(d_keys, keys.data(), total_ops * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 3. 启动 Kernel
    // Shared Memory 大小 = capacity * sizeof(uint32_t)
    size_t smem_size = capacity * sizeof(uint32_t);
    
    // 这里的关键：虽然我们是并发插入，但我们要验证结果的一致性。
    // 为了简化验证，我们可以先不做全并发乱序插入，而是由线程按顺序领任务。
    // 不过 atomicCAS 本身就是处理乱序的。我们只要统计总的成功数即可。
    
    test_hashmap_kernel<<<1, 256, smem_size>>>(
        bitlen, 
        total_ops, 
        d_keys, 
        d_results, 
        d_debug_table
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 验证结果
    std::vector<uint32_t> h_results(total_ops);
    std::vector<uint32_t> h_debug_table(capacity);
    CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, total_ops * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_debug_table.data(), d_debug_table, capacity * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 统计
    int success_count = 0;
    int fail_count = 0;
    for (int r : h_results) {
        if (r == 1) success_count++;
        else fail_count++;
    }

    std::cout << "   Actual Success: " << success_count << ", Actual Fail: " << fail_count << std::endl;

    // 验证逻辑 1: 成功的次数应该等于唯一 Key 的数量 (除非表满了)
    if (num_unique_items <= capacity) {
        if (success_count != num_unique_items) {
            std::cout << "   [FAIL] Expected " << num_unique_items << " successes, got " << success_count << std::endl;
            exit(1);
        }
    }

    // 验证逻辑 2: 检查 Hashmap 内部数据
    // 统计表中非 INVALID 的数量，应该等于 success_count
    int table_entries = 0;
    for (uint32_t val : h_debug_table) {
        if (val != 0xFFFFFFFF) { // INVALID_KEY
            table_entries++;
        }
    }

    if (table_entries != success_count) {
        std::cout << "   [FAIL] Table entries (" << table_entries << ") != Success count (" << success_count << ")" << std::endl;
        exit(1);
    }

    std::cout << "   [PASS] Functionality verified." << std::endl;

    CHECK_CUDA(cudaFree(d_keys));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_debug_table));
}

int main() {
    // 1. 基础功能测试：Bitlen=10 (Cap=1024), 插入 500 个唯一，200 个重复
    run_test(10, 500, 0.4); 

    std::cout << "---------------------------------------------------" << std::endl;

    // 2. 高负载测试：Bitlen=8 (Cap=256), 插入 200 个唯一 (Load ~80%)
    // 这会大量触发线性探测 (Linear Probing)
    run_test(8, 200, 0.0);

    std::cout << "---------------------------------------------------" << std::endl;

    // 3. 满载测试 (Stress Test): Bitlen=8 (Cap=256), 插入 256 个唯一
    // 理论上应该都能插进去，但性能会下降
    run_test(8, 256, 0.0);

    return 0;
}