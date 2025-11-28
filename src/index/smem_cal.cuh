#pragma once
#include "config.cuh"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace cagra {
namespace detail {

// 辅助函数：计算大于等于 n 的最小 2 的幂次
// 例如: 10 -> 16, 300 -> 512
inline uint32_t next_power_of_2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/**
 * @brief 计算并检查 Shared Memory 需求
 */
inline size_t calculate_and_check_smem(uint32_t itopk_size, 
                                       uint32_t search_width, 
                                       uint32_t graph_degree) 
{
    using namespace config;
    size_t smem = 0;

    // 1. Query Buffer [1024 * 4 bytes] = 4KB
    // 必须 16 字节对齐
    size_t query_size = DIM * sizeof(float);
    smem += (query_size + 15) & ~15;

    // 2. Visited Hashmap [4096 * 4 bytes] = 16KB
    size_t hash_size = (1u << DEFAULT_HASH_BITLEN) * sizeof(uint32_t);
    smem += (hash_size + 15) & ~15;

    // 3. 结果队列 & 候选队列 (关键修改点)
    // 实际需要的元素数量
    uint32_t raw_needed = itopk_size + search_width * graph_degree;
    uint32_t queue_capacity = std::max(BLOCK_SIZE, next_power_of_2(raw_needed));
    size_t queue_mem = queue_capacity * (sizeof(uint32_t) + sizeof(float));
    smem += (queue_mem + 15) & ~15;

    // 4. Parent List (下一轮要扩展的节点 ID)
    size_t parent_raw_size = search_width * sizeof(uint32_t);
    size_t parent_mem = (parent_raw_size + 15) & ~15; // 向上取整到 16 字节
    smem += parent_mem;

    // 5. 杂项
    smem += 256;

    // ==========================================
    // 核心检查逻辑
    // ==========================================
    if (smem > MAX_SHARED_MEMORY) {
        std::cerr << "Error: Shared Memory limit exceeded!" << std::endl;
        std::cerr << "  User Request: itopk=" << itopk_size << std::endl;
        std::cerr << "  Queue Cap: " << queue_capacity << " (aligned to pow2)" << std::endl;
        std::cerr << "  Total SMEM: " << smem << " bytes" << std::endl;
        std::cerr << "  Hardware Limit: " << MAX_SHARED_MEMORY << " bytes" << std::endl;
        throw std::runtime_error("Shared Memory limit exceeded");
    }

    return smem;
}

} // namespace detail
} // namespace cagra