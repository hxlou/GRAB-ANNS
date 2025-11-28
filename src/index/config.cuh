#pragma once
#include <cstdint>

namespace cagra {
namespace config {

// ==========================================
// 全局硬件约束 (Global Constraints)
// ==========================================

// 1. 限制最大 Shared Memory 使用量 (48KB)
// 这保证了 L1 Cache 有 80KB (Ampere架构)，性能最优
constexpr size_t MAX_SHARED_MEMORY = 48 * 1024; 

// 2. 固定硬件参数 (针对 1024 维优化)
constexpr uint32_t DIM = 1024;
constexpr uint32_t TEAM_SIZE = 32;       // Warp Size
constexpr uint32_t BLOCK_SIZE = 256;     // 推荐 Block 大小 (8 Warps)

// 3. 算法参数默认值
constexpr uint32_t DEFAULT_HASH_BITLEN = 12; // Hashmap 大小 4096

} // namespace config
} // namespace cagra