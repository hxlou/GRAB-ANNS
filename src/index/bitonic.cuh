#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cagra {
namespace bitonic {

// ==========================================
// 1. 基础交换函数
// ==========================================

// 线程内：寄存器交换
template <class K, class V>
__device__ __forceinline__ void swap_local(K& kA, V& vA, K& kB, V& vB, bool want_min_at_A) {
    // 如果 A 想要最小值，但 A > B，则交换
    // 如果 A 想要最大值，但 A < B，则交换
    // 逻辑归一化：(kA < kB) != want_min_at_A
    if ((kA != kB) && ((kA < kB) != want_min_at_A)) {
        K tK = kA; kA = kB; kB = tK;
        V tV = vA; vA = vB; vB = tV;
    }
}

// 跨线程：Shuffle 交换
template <class K, class V>
__device__ __forceinline__ void swap_peer(K& k, V& v, uint32_t mask, uint32_t peer_lane, bool want_min) {
    K other_k = __shfl_sync(mask, k, peer_lane);
    V other_v = __shfl_sync(mask, v, peer_lane);

    // 比较我(k)和对方(other_k)
    // 只有当我的持有值不符合我的期望(want_min)时，才更新
    // 例如：我想要 min，但我比对方大 (k > other_k)，那我应该拿对方的值
    if ((k != other_k) && ((k < other_k) != want_min)) {
        k = other_k;
        v = other_v;
    }
}

// ==========================================
// 2. 统一逻辑的双调排序
// ==========================================

/**
 * @brief Warp 级双调排序 (Unified Indexing Approach)
 * 
 * 将 Warp 内 32 个线程持有的共 32*N 个元素视为一个整体进行排序。
 * 
 * @tparam N 每个线程持有的元素数量 (必须是 2 的幂次: 2, 4, 8...)
 * @param k  Key 数组 (寄存器)
 * @param v  Value 数组 (寄存器)
 * @param asc true=全局升序, false=全局降序
 */
template <class K, class V, int N>
__device__ __forceinline__ void warp_sort(K k[N], V v[N], bool asc = true) {
    const int lane_id = threadIdx.x % 32;
    const int total_elems = 32 * N;

    // -------------------------------------------------------
    // 双调排序主循环
    // size: 当前构建的双调序列长度 (2, 4, 8 ... total_elems)
    // -------------------------------------------------------
    for (int size = 2; size <= total_elems; size <<= 1) {
        
        // 确定当前线程持有的 N 个数据，相对于 size 块的方向
        // 但由于 N 可能大于 1，每个元素的方向可能不同，所以我们在内层循环判断
        
        // stride: 比较跨度 (size/2, size/4 ... 1)
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            
            // 遍历当前线程持有的 N 个元素
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                // 计算当前元素的全局索引 [0, 32*N - 1]
                int global_idx = lane_id * N + i;
                
                // 计算比较对手的全局索引
                int partner_idx = global_idx ^ stride;

                // 只有当 global_idx < partner_idx 时才主动发起交换逻辑
                // (对于跨线程，双方都会计算，我们通过 want_min 控制谁拿谁)
                
                // 1. 确定当前元素所属的 size 块的方向
                // ((global_idx & size) == 0) 表示前半段
                // asc == true:  前半段(0)升序(T), 后半段(1)降序(F) -> Match
                // asc == false: 前半段(0)降序(F), 后半段(1)升序(T) -> Match
                bool direction = ((global_idx & size) == 0) != !asc;

                // 2. 确定当前位置 (global_idx) 是否应该持有较小值
                // 如果是升序块 (direction=T)，较小索引 (global_idx) 拿小值
                // 如果是降序块 (direction=F)，较小索引 (global_idx) 拿大值
                // 这里的逻辑简化为：
                bool want_min = (global_idx < partner_idx) == direction;

                // 3. 执行交换
                if (stride >= N) {
                    // Case A: 对手在另一个线程
                    int peer_lane = lane_id ^ (stride / N);
                    swap_peer(k[i], v[i], 0xFFFFFFFF, peer_lane, want_min);
                } else {
                    // Case B: 对手在同一个线程
                    // 只有当我是较小索引时才执行(避免重复交换)
                    // 线程内 partner_idx 对应的寄存器下标是 (i ^ stride)
                    int j = i ^ stride;
                    if (i < j) {
                        swap_local(k[i], v[i], k[j], v[j], want_min);
                    }
                }
            }
        }
    }
}

} // namespace bitonic
} // namespace cagra