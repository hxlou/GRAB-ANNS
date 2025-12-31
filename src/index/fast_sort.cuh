#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace cagra {
namespace fast_sort {

// =========================================================
// 1. 基础工具：比较交换指令 (CAS)
// =========================================================

// 本地寄存器交换 (Local CAS)
template <typename K, typename V>
__device__ __forceinline__ void compare_and_swap(K& kA, V& vA, K& kB, V& vB, bool dir) {
    // dir=true (Asc): A < B -> Keep A. 
    // Logic: if ((kA > kB) == dir) swap
    // 这种逻辑会被编译成高效的 FMIN/FMAX 指令序列
    if ((kA > kB) == dir) {
        K tK = kA; kA = kB; kB = tK;
        V tV = vA; vA = vB; vB = tV;
    }
}

// 跨线程交换 (Shuffle CAS)
template <typename K, typename V>
__device__ __forceinline__ void compare_and_exchange_peer(K& k, V& v, int mask, bool dir) {
    K other_k = __shfl_xor_sync(0xFFFFFFFF, k, mask);
    V other_v = __shfl_xor_sync(0xFFFFFFFF, v, mask);
    
    // 判断当前线程是 pair 中的 Low 还是 High
    // 如果 mask=1 (0-1), lane 0 是 low, lane 1 是 high
    bool am_i_lower = (threadIdx.x & mask) == 0;
    
    // 如果 dir=Asc(true): 
    //   Lower lane (0) 想要 min
    //   Higher lane (1) 想要 max
    // 归一化条件: 我是否应该保留较小值?
    bool want_min = (dir == am_i_lower); 

    // 如果 (k > other) 且 我想要 min -> 交换 (拿 other)
    // 如果 (k < other) 且 我想要 max -> 交换 (拿 other)
    if ((k > other_k) == want_min) {
        k = other_k;
        v = other_v;
    }
}

// =========================================================
// 2. 递归模板：完全展开的排序网络
// =========================================================

/**
 * 核心逻辑：Bitonic Merge Step
 * 对应标准算法中内层的 stride 循环
 * 
 * SIZE:   当前双调序列的大小 (2, 4, 8 ...)
 * STRIDE: 当前比较跨度 (SIZE/2, SIZE/4 ... 1)
 * N:      每个线程持有的元素数量
 */
template <int SIZE, int STRIDE, int N>
struct BitonicMergeStep {
    __device__ __forceinline__ static void run(float key[N], uint32_t val[N]) {
        int lane_id = threadIdx.x % 32;

        // --- 分支 A: 跨线程通信 (Shuffle) ---
        if constexpr (STRIDE >= N) {
            int peer_mask = STRIDE / N;
            
            // 计算方向：Standard Bitonic Sort Direction Logic
            // ((global_idx / SIZE) % 2 == 0) -> Ascending
            // global_idx = lane_id * N + i
            // 由于 N 是 2 的幂，且 STRIDE >= N，
            // (lane_id * N + i) & SIZE 只取决于 lane_id
            bool dir = ((lane_id * N) & SIZE) == 0;
            
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                compare_and_exchange_peer(key[i], val[i], peer_mask, dir);
            }
        } 
        // --- 分支 B: 本地寄存器操作 (Local) ---
        else {
            // 此时 STRIDE < N，操作都在 key 数组内部
            // 这里的 i 和 i^STRIDE 都是编译期常量，不会触发 Local Memory
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                int partner = i ^ STRIDE;
                if (i < partner) {
                    // 重新计算精确的方向
                    // 当 SIZE > N 时，方向由 lane_id 决定 (整块方向一致)
                    // 当 SIZE <= N 时，方向由 i 决定 (块内交错)
                    int global_idx = lane_id * N + i;
                    bool dir = (global_idx & SIZE) == 0;
                    
                    compare_and_swap(key[i], val[i], key[partner], val[partner], dir);
                }
            }
        }

        // 递归调用下一个更小的 STRIDE
        BitonicMergeStep<SIZE, STRIDE / 2, N>::run(key, val);
    }
};

// 递归终止条件 (STRIDE = 0)
template <int SIZE, int N>
struct BitonicMergeStep<SIZE, 0, N> {
    __device__ __forceinline__ static void run(float key[N], uint32_t val[N]) {}
};

/**
 * 外层逻辑：Bitonic Sort Stage
 * 对应标准算法中外层的 size 循环
 * 
 * SIZE: 当前构建的双调序列大小 (2, 4 ... 32*N)
 */
template <int SIZE, int N>
struct BitonicSortStage {
    __device__ __forceinline__ static void run(float key[N], uint32_t val[N]) {
        // 1. 处理更小的 stage
        BitonicSortStage<SIZE / 2, N>::run(key, val);

        // 2. 执行当前的 Merge
        // 初始 stride 为 SIZE / 2
        BitonicMergeStep<SIZE, SIZE / 2, N>::run(key, val);
    }
};

// 递归终止条件 (SIZE = 1)
template <int N>
struct BitonicSortStage<1, N> {
    __device__ __forceinline__ static void run(float key[N], uint32_t val[N]) {}
};



// =========================================================
// 3. 你的顶层接口实现
// =========================================================

template <int N> // N = Capacity / 32
__device__ __forceinline__ void load_sort_store(float* smem_dists, uint32_t* smem_indices, uint32_t capacity) {
    // 1. 定义寄存器
    float key[N];
    uint32_t val[N];
    int lane_id = threadIdx.x; // 0-31

    // 2. 从 Shared Memory 加载到寄存器 (Blocked Layout)
    // 必须做边界检查并填充 FLT_MAX，否则无效数据会参与排序并排到前面
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id * N + i; 
        if (idx < capacity) {
            key[i] = smem_dists[idx];
            val[i] = smem_indices[idx];
        } else {
            key[i] = FLT_MAX;     // 无穷大排到最后
            val[i] = 0xFFFFFFFF;  // 无效索引
        }
    }

    // 3. 执行完全展开的双调排序
    // 这里会生成几千行高效的汇编代码，无循环，无 Local Memory
    // 排序范围是 N * 32 (即 Capacity 的上限)
    cagra::fast_sort::BitonicSortStage<N * 32, N>::run(key, val);

    // 4. 写回 Shared Memory
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id * N + i;
        if (idx < capacity) {
            smem_dists[idx] = key[i];
            smem_indices[idx] = val[i];
        }
    }
}

} // namespace bitonic
} // namespace cagra
