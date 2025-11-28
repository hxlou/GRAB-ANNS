#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cagra {
namespace hashmap {

// 定义无效 Key 值 (类似于 NULL)
constexpr uint32_t INVALID_KEY = 0xFFFFFFFF;

// 计算 Hash 表大小 (2^bitlen)
__device__ __forceinline__ uint32_t compute_size(const uint32_t bitlen) { 
    return 1u << bitlen; 
}

/**
 * @brief 初始化 Hash 表
 * 将所有位置置为 INVALID_KEY。
 * 支持多线程并行初始化。
 * 
 * @param table   共享内存中的 Hash 表指针
 * @param bitlen  Hash 表大小的对数 (例如 10 表示大小 1024)
 */
__device__ __forceinline__ void init(uint32_t* table, const uint32_t bitlen) {
    uint32_t size = compute_size(bitlen);
    // 典型的 Grid-Stride Loop 写法，让当前 Block 的所有线程协作初始化
    for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
        table[i] = INVALID_KEY;
    }
}

/**
 * @brief 简单的 Hash 函数
 * @param key     节点 ID
 * @param bitlen  Hash 表大小的对数
 */
__device__ __forceinline__ uint32_t hash_func(uint32_t key, uint32_t bitlen) {
    // 原始 RAFT 的 Hash 策略：异或高位，打散分布
    return (key ^ (key >> bitlen));
}

/**
 * @brief 插入 Key (核心函数)
 * 
 * 尝试将 Key 插入 Hash 表。
 * 如果 Key 已存在，返回 0 (false)。
 * 如果 Key 不存在且插入成功，返回 1 (true)。
 * 
 * @param table   Hash 表指针
 * @param bitlen  Hash 表大小的对数
 * @param key     要插入的节点 ID
 */
__device__ __forceinline__ bool insert(uint32_t* table, const uint32_t bitlen, const uint32_t key) {
    const uint32_t size = compute_size(bitlen);
    const uint32_t mask = size - 1;
    
    // 1. 计算初始索引
    uint32_t index = hash_func(key, bitlen) & mask;
    
    // 2. 线性探测 (Linear Probing)
    // 最坏情况遍历整个表，但在合理负载因子下通常只需几次
    for (uint32_t i = 0; i < size; i++) {
        
        // 3. 原子比较并交换 (CAS)
        // 尝试把 INVALID_KEY 修改为 key
        // atomicCAS 返回该地址原来的值
        uint32_t old = atomicCAS(&table[index], INVALID_KEY, key);
        
        // 情况 A: 原来是 INVALID_KEY，说明位置是空的，我们抢到了 -> 插入成功
        if (old == INVALID_KEY) {
            return true;
        }
        // 情况 B: 原来就是 key，说明之前已经有人插过了 -> 已存在
        else if (old == key) {
            return false;
        }
        
        // 情况 C: 原来是别的 key (冲突)，继续探测下一个位置
        // (index + 1) & mask 实现了循环数组
        index = (index + 1) & mask;
    }
    
    // 表满了 (理论上在 CAGRA 搜索中不应该发生，因为我们会控制 visited 数量小于表大小)
    return false; 
}

} // namespace hashmap
} // namespace cagra