#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// 保持引用系统自带的 CUB
#include <cub/warp/warp_merge_sort.cuh>

namespace cagra {
namespace merge {
// =========================================================
// 1. 定义一个简单的比较器 (Functor)
// =========================================================
// CUB 需要一个结构体，重载 operator() 来告诉它怎么比大小
struct FloatLess {
    __device__ __forceinline__ bool operator()(const float& a, const float& b) const {
        // 处理 NaN 的严谨写法，或者简单 return a < b;
        return a < b;
    }
};

// =========================================================
// 2. 基于 CUB WarpMergeSort 的封装
// =========================================================

/**
 * @tparam N 每个线程持有的元素个数
 */
template <int N>
__device__ __forceinline__ void load_sort_store(float* smem_dists, uint32_t* smem_indices, uint32_t capacity) {
    
    // 定义 CUB WarpMergeSort 类型
    // <KeyT, ITEMS_PER_THREAD, WARP_SZ, ValueT>
    using WarpSortT = cub::WarpMergeSort<float, N, 32, uint32_t>;

    // 准备寄存器
    float keys[N];
    uint32_t values[N];
    int lane_id = threadIdx.x % 32;

    // Load
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id * N + i;
        if (idx < capacity) {
            keys[i] = smem_dists[idx];
            values[i] = smem_indices[idx];
        } else {
            keys[i] = FLT_MAX;      
            values[i] = 0xFFFFFFFF; 
        }
    }

    // Temp Storage
    typename WarpSortT::TempStorage* temp_storage = 
        reinterpret_cast<typename WarpSortT::TempStorage*>(smem_dists);

    // =====================================================
    // 【关键修复】显式传入比较器 FloatLess()
    // =====================================================
    // 强制调用 Sort(keys, values, compare_op) 重载
    // 这样编译器就不会把 values 误认为是 compare_op 了
    WarpSortT(*temp_storage).Sort(keys, values, FloatLess());

    // Store
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int idx = lane_id * N + i;
        if (idx < capacity) {
            smem_dists[idx] = keys[i];
            smem_indices[idx] = values[i];
        }
    }
}

} // namespace merge
} // namespace cagra