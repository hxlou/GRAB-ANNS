#pragma once

#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>

namespace cagra {
namespace multi_cta_sort {

template <class K, class V>
__device__ __forceinline__ void swap_if_needed(K& k0, V& v0, K& k1, V& v1, bool asc)
{
    if ((k0 != k1) && ((k0 < k1) != asc)) {
        K tk = k0;
        V tv = v0;
        k0 = k1;
        v0 = v1;
        k1 = tk;
        v1 = tv;
    }
}

template <class K, class V>
__device__ __forceinline__ void swap_if_needed(K& k, V& v, uint32_t lane_offset, bool asc)
{
    K other_k = __shfl_xor_sync(0xffffffffu, k, lane_offset);
    V other_v = __shfl_xor_sync(0xffffffffu, v, lane_offset);
    if ((k != other_k) && ((k < other_k) != asc)) {
        k = other_k;
        v = other_v;
    }
}

template <class K, class V, uint32_t N>
__device__ __forceinline__ void warp_merge(K (&key)[N],
                                           V (&val)[N],
                                           uint32_t range,
                                           bool asc = true)
{
    const uint32_t lane_id = threadIdx.x & 31u;

    if (range == 1) {
        for (uint32_t b = 2; b <= N; b <<= 1) {
            for (uint32_t c = b >> 1; c >= 1; c >>= 1) {
                #pragma unroll
                for (uint32_t i = 0; i < N; ++i) {
                    uint32_t j = i ^ c;
                    if (i >= j) continue;
                    const uint32_t line_id = i + N * lane_id;
                    bool direction = static_cast<bool>(line_id & b) ==
                                     static_cast<bool>(line_id & c);
                    swap_if_needed(key[i], val[i], key[j], val[j], direction);
                }
                if (c == 1) break;
            }
        }
        return;
    }

    const uint32_t b = range;
    for (uint32_t c = b >> 1; c >= 1; c >>= 1) {
        bool direction = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
        #pragma unroll
        for (uint32_t i = 0; i < N; ++i) {
            swap_if_needed(key[i], val[i], c, direction);
        }
        if (c == 1) break;
    }

    bool direction = ((lane_id & b) == 0);
    for (uint32_t c = N >> 1; c >= 1; c >>= 1) {
        #pragma unroll
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t j = i ^ c;
            if (i >= j) continue;
            swap_if_needed(key[i], val[i], key[j], val[j], direction);
        }
        if (c == 1) break;
    }
}

template <class K, class V, uint32_t N>
__device__ __forceinline__ void warp_sort(K (&key)[N], V (&val)[N], bool asc = true)
{
    #pragma unroll
    for (uint32_t range = 1; range <= 32; range <<= 1) {
        warp_merge<K, V, N>(key, val, range, asc);
    }
}

template <uint32_t N>
__device__ __forceinline__ void load_sort_store(float* smem_dists,
                                                uint32_t* smem_indices,
                                                uint32_t capacity)
{
    static_assert(N == 2 || N == 4 || N == 8 || N == 16,
                  "multi_cta_sort supports 64/128/256/512 element buffers");
    const uint32_t lane_id = threadIdx.x & 31u;
    float key[N];
    uint32_t val[N];

    #pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t idx = lane_id + 32u * i;
        if (idx < capacity) {
            key[i] = smem_dists[idx];
            val[i] = smem_indices[idx];
        } else {
            key[i] = FLT_MAX;
            val[i] = 0xffffffffu;
        }
    }

    warp_sort<float, uint32_t, N>(key, val, true);

    #pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t idx = N * lane_id + i;
        if (idx < capacity) {
            smem_dists[idx] = key[i];
            smem_indices[idx] = val[i];
        }
    }
}

} // namespace multi_cta_sort
} // namespace cagra
