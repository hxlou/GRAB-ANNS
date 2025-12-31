#include <vector>
#include <omp.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp> // 确保包含这个用于创建 view
#include <raft/neighbors/brute_force.cuh>

// 定义每组数据的输入输出结构
struct KnnGroupTask {
    const float* data_ptr; 
    int64_t rows;
    int64_t cols;
    
    int* out_indices_ptr;
    float* out_distances_ptr;

    int k;
};

void run_batch_knn(const raft::device_resources& main_res, 
                   const std::vector<KnnGroupTask>& tasks) {
    
    int num_tasks = tasks.size();

    // OpenMP 并行提交任务
    // 这里的 num_threads 可以控制并发度，比如 num_threads(8)
    #pragma omp parallel for schedule(dynamic) 
    for (int i = 0; i < num_tasks; ++i) {
        const auto& task = tasks[i];

        // 1. 创建独立的 Stream
        cudaStream_t stream;
        // 使用 cudaStreamNonBlocking 以获得最佳并发性能
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        // 2. 创建绑定到该 Stream 的 RAFT 句柄
        raft::device_resources local_res(stream);

        // 3. 创建数据视图 (View) - 零拷贝
        // 注意：维度必须是 int64_t
        auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
            task.data_ptr, task.rows, task.cols
        );

        auto indices_view = raft::make_device_matrix_view<int, int64_t>(
            task.out_indices_ptr, task.rows, task.k
        );

        auto dists_view = raft::make_device_matrix_view<float, int64_t>(
            task.out_distances_ptr, task.rows, task.k
        );

        // 4. Build 阶段 (构建索引)
        // 对于 brute_force，这里非常快，几乎没有开销
        auto index = raft::neighbors::brute_force::build(local_res, dataset_view);

        // 5. Search 阶段
        // 将 dataset_view 同时作为查询数据 (queries)
        raft::neighbors::brute_force::search(
            local_res,
            index,         // 刚才构建的索引
            dataset_view,  // 查询数据 (Self-Join)
            indices_view,  // 输出索引
            dists_view     // 输出距离
        );

        // 6. 同步并销毁流
        // 必须同步，否则销毁流会导致未完成的任务被杀掉
        local_res.sync_stream();
        cudaStreamDestroy(stream);
    }
}


inline void build_batch_knn_graphs(
    const float* d_dataset,
    int dim,
    const std::vector<size_t>& bucket_sizes,
    int k,
    uint32_t* d_graph
) {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::set_current_device_resource(&cuda_mr);
    // 1. 预计算每个 bucket 在 d_dataset 和 d_graph 中的偏移量
    //    这是为了避免在 OpenMP 并行区中进行由于依赖导致的串行计算
    size_t num_buckets = bucket_sizes.size();
    std::vector<size_t> data_offsets(num_buckets, 0);
    std::vector<size_t> graph_offsets(num_buckets, 0);

    size_t current_data_offset = 0;
    size_t current_graph_offset = 0;

    for (size_t i = 0; i < num_buckets; ++i) {
        data_offsets[i] = current_data_offset;
        graph_offsets[i] = current_graph_offset;

        // 更新偏移量
        current_data_offset += bucket_sizes[i] * static_cast<size_t>(dim);
        current_graph_offset += bucket_sizes[i] * static_cast<size_t>(k);
    }

    // 2. OpenMP 并行提交 CUDA 任务
    //    schedule(dynamic) 允许处理完小 bucket 的线程抢占下一个任务，负载均衡
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_buckets; ++i) {
        size_t n_rows = bucket_sizes[i];
        
        // 极小 bucket 保护（如果 bucket 小于 k，RAFT 可能会报错或行为未定义，视具体需求处理）
        if (n_rows == 0) continue; 

        // -----------------------------------------------------------
        // A. 资源隔离
        // -----------------------------------------------------------
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        raft::device_resources local_res(stream);
        rmm::cuda_stream_view stream_view(stream);

        // -----------------------------------------------------------
        // B. 指针计算
        // -----------------------------------------------------------
        const float* curr_data_ptr = d_dataset + data_offsets[i];
        uint32_t* curr_graph_ptr = d_graph + graph_offsets[i];
        uint32_t current_graph_offset = graph_offsets[i];

        // -----------------------------------------------------------
        // C. 创建视图 (Views)
        // -----------------------------------------------------------
        
        // 输入数据视图
        auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
            curr_data_ptr, n_rows, dim
        );

        // 输出索引视图 (uint32_t -> int 转换)
        // RAFT 接口要求 int (int32)，但二进制表示与 uint32 一致。
        // reinterpret_cast 在这里是安全的，实现了零拷贝直接写入 d_graph。
        auto indices_view = raft::make_device_matrix_view<int, int64_t>(
            reinterpret_cast<int*>(curr_graph_ptr), n_rows, k
        );

        // 临时距离缓冲区 (RAFT 要求必须提供)
        // 使用 RMM 快速分配，任务结束后自动释放
        rmm::device_uvector<float> temp_dists(n_rows * k, stream_view);
        auto dists_view = raft::make_device_matrix_view<float, int64_t>(
            temp_dists.data(), n_rows, k
        );

        // -----------------------------------------------------------
        // D. 核心计算 (Build + Search)
        // -----------------------------------------------------------
        
        // 构建索引 (对于 Brute Force，这是一个极轻量的操作)
        auto index = raft::neighbors::brute_force::build(local_res, dataset_view);

        // 搜索 (Self-Join: 自己搜自己，构建 KNN 图)
        raft::neighbors::brute_force::search(
            local_res,
            index,          // 索引
            dataset_view,   // 查询向量 (同索引向量)
            indices_view,   // [输出] 写入用户的 d_graph
            dists_view      // [临时] 丢弃
        );

        // -----------------------------------------------------------
        // E. 清理
        // -----------------------------------------------------------
        // 必须同步，因为 temp_dists 析构时需要确保 GPU 已经用完了这块内存
        // 且需要确保 stream 销毁前任务已完成
        local_res.sync_stream();
        cudaStreamDestroy(stream);
    }
}