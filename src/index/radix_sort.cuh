#include <cub/cub.cuh>
#include <cfloat>

namespace cagra {
namespace radix {

// 设定最大支持的处理能力
// 16 * 256(线程) = 4096 容量。
// 只要你的 capacity <= 4096，这个函数都能处理。
#define MAX_ITEMS_PER_THREAD 4

/**
 * 统一版排序函数
 * 接口保持不变，无需传入外部临时空间指针
 */
__device__ __forceinline__ void load_sort_store(
    float* smem_dists, 
    uint32_t* smem_indices, 
    uint32_t capacity
) {
    // 1. 定义类型：固定使用 MAX_ITEMS_PER_THREAD
    using BlockSortT = cub::BlockRadixSort<float, 256, MAX_ITEMS_PER_THREAD, uint32_t>;

    // 2. 【关键】静态自动分配临时空间
    // 编译器会自动计算大小并在 Kernel 启动时预留，不需要你操心
    // 因为这里只有一种 BlockSortT，所以只会占用一份空间 (约 6KB)，不会溢出
    __shared__ typename BlockSortT::TempStorage temp_storage;

    // 3. 准备寄存器 (始终分配 16 个，哪怕只排 64 个数)
    float keys[MAX_ITEMS_PER_THREAD];
    uint32_t values[MAX_ITEMS_PER_THREAD];
    
    int lane_id = threadIdx.x; // 假设 blockDim=256

    // 4. Load & Padding (加载与填充)
    #pragma unroll
    for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i) {
        int idx = lane_id * MAX_ITEMS_PER_THREAD + i;
        
        // 只有在有效范围内才加载数据，否则填充最大值 (Padding)
        if (idx < capacity) {
            keys[i] = smem_dists[idx];
            values[i] = smem_indices[idx];
        } else {
            // 填充 FLT_MAX，这样它们排序后会自动跑到底部
            keys[i] = FLT_MAX;      
            values[i] = 0xFFFFFFFF; 
        }
    }

    // 5. 排序 (直接传入对象引用，不要加 *)
    BlockSortT(temp_storage).Sort(keys, values);

    // 6. Store (写回)
    #pragma unroll
    for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i) {
        int idx = lane_id * MAX_ITEMS_PER_THREAD + i;
        
        // 只把有效范围内的数据写回去
        if (idx < capacity) {
            smem_dists[idx] = keys[i];
            smem_indices[idx] = values[i];
        }
    }
}

} // namespace radix
} // namespace cagra