#include "../src/cagraIndex.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cassert>
#include <memory>
#include <cstring>

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS (仅用于生成真值 GT)
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/IndexFlat.h> // 引入 CPU FAISS 头文件


// -----------------------------------------------------------------------------
// 辅助工具
// -----------------------------------------------------------------------------

template<typename T>
void safe_delete(T*& ptr) {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

// 加载fvecs文件 (标准fvecs格式：每个向量以int维度开头)
std::vector<float> load_fvecs(const std::string& filename, int& dim, int& num_vectors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return {};
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < sizeof(int)) {
        std::cerr << "Error: File too small to contain dimension info: " << filename << std::endl;
        return {};
    }

    // 读取第一个向量的维度
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (file.gcount() != sizeof(int)) {
        std::cerr << "Error: Failed to read dimension from: " << filename << std::endl;
        return {};
    }

    // 标准fvecs格式：每个向量 = sizeof(int) + dim * sizeof(float)
    size_t vector_size_bytes = sizeof(int) + dim * sizeof(float);
    num_vectors = file_size / vector_size_bytes;

    std::cout << "File size: " << file_size << " bytes, vector size: " << vector_size_bytes << " bytes" << std::endl;
    std::cout << "Loading " << num_vectors << " vectors of dimension " << dim << " from " << filename << std::endl;

    // 重新定位到文件开头
    file.seekg(0, std::ios::beg);

    std::vector<float> data(num_vectors * dim);

    // 逐个向量读取 (跳过每个向量的维度头)
    for (size_t i = 0; i < num_vectors; ++i) {
        int vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int));

        if (vec_dim != dim) {
            std::cerr << "Error: Vector " << i << " has dimension " << vec_dim << ", expected " << dim << std::endl;
            return {};
        }

        file.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
        if (file.gcount() != dim * sizeof(float)) {
            std::cerr << "Error: Failed to read vector " << i << " data" << std::endl;
            return {};
        }
    }

    return data;
}

// 生成ground truth (CPU FAISS) - 增加内存检查和错误处理
void generate_ground_truth(const float* dataset, size_t n_dataset, int dim,
                            const float* queries, size_t n_queries, int k,
                            int64_t* gt_indices) {
    try {
        std::cout << "[GT] Generating ground truth for " << n_queries << " queries in " << n_dataset << " vectors..." << std::endl;
        std::cout << "[GT] Dataset size: " << (n_dataset * dim * sizeof(float) / 1024 / 1024) << " MB" << std::endl;
        std::cout << "[GT] Queries size: " << (n_queries * dim * sizeof(float) / 1024) << " KB" << std::endl;

        // 检查指针有效性
        if (dataset == nullptr) {
            std::cerr << "[GT] Error: dataset pointer is null" << std::endl;
            throw std::runtime_error("dataset pointer is null");
        }
        if (queries == nullptr) {
            std::cerr << "[GT] Error: queries pointer is null" << std::endl;
            throw std::runtime_error("queries pointer is null");
        }
        if (gt_indices == nullptr) {
            std::cerr << "[GT] Error: gt_indices pointer is null" << std::endl;
            throw std::runtime_error("gt_indices pointer is null");
        }

        std::cout << "[GT] Creating FAISS index..." << std::endl;
        faiss::IndexFlatL2 cpu_index(dim);

        std::cout << "[GT] Adding vectors to FAISS index..." << std::endl;
        cpu_index.add(n_dataset, dataset);
        std::cout << "[GT] Successfully added " << n_dataset << " vectors" << std::endl;

        // 检查数据有效性 - 先测试一个小搜索
        std::cout << "[GT] Testing with single query first..." << std::endl;
        std::vector<float> distances(k);
        try {
            cpu_index.search(1, queries, k, distances.data(), gt_indices);
            std::cout << "[GT] Single query test successful." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[GT] Single query test failed: " << e.what() << std::endl;
            throw;
        }

        std::cout << "[GT] Checking query data validity..." << std::endl;
        // 检查查询数据是否有无效值
        for (size_t q = 0; q < n_queries; ++q) {
            bool has_nan = false, has_inf = false;
            for (int d = 0; d < dim; ++d) {
                float val = queries[q * dim + d];
                if (std::isnan(val)) has_nan = true;
                if (std::isinf(val)) has_inf = true;
            }
            if (has_nan || has_inf) {
                std::cout << "[GT] Query " << q << " has NaN: " << has_nan << ", Inf: " << has_inf << std::endl;
            }
        }

        std::cout << "[GT] Full search started (batch processing)..." << std::endl;
        // 分批处理查询以避免内存问题，提供有效的距离数组
        const int batch_size = 1000;  // 每批处理1000个查询，提高GPU利用率
        std::vector<float> batch_distances(batch_size * k);  // 为每批分配距离数组

        for (size_t batch_start = 0; batch_start < n_queries; batch_start += batch_size) {
            size_t current_batch_size = std::min(batch_size, static_cast<int>(n_queries - batch_start));
            std::cout << "[GT] Processing batch " << (batch_start / batch_size + 1)
                      << " (" << current_batch_size << " queries)..." << std::endl;

            cpu_index.search(
                current_batch_size,
                queries + batch_start * dim,
                k,
                batch_distances.data(),  // 提供有效的距离数组，不是nullptr
                gt_indices + batch_start * k
            );
        }
        std::cout << "[GT] Ground truth completed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[GT] Error generating ground truth: " << e.what() << std::endl;
        throw;
    }
}

// 计算召回率
double calculate_recall(int n_queries, int k, const int64_t* gt, const int64_t* result) {
    int total_hits = 0;
    for (int i = 0; i < n_queries; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) {
            gt_set.insert(gt[i * k + j]);
        }
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(result[i * k + j])) {
                total_hits++;
            }
        }
    }
    return static_cast<double>(total_hits) / (n_queries * k);
}

// =============================================================================
// 测试函数
// =============================================================================

// 测试: GT生成 -> Add -> Build -> Query -> Save
void test_build_query(const float* base_data, int dim, int n_base,
                     const float* query_data, int n_query, const std::string& index_path) {
    std::cout << "\n[TEST] Build + Query Test" << std::endl;
    const size_t N_BUILD = n_base;  // 使用全部数据
    const int K = 10;
    const size_t N_QUERY = n_query;   // 使用全部查询

    // 1. 先生成GT (避免构建完CAGRA后GT失败)
    std::cout << ">> [Step 1] Generating ground truth with CPU FAISS..." << std::endl;
    std::vector<int64_t> gt_indices(N_QUERY * K);
    generate_ground_truth(base_data, N_BUILD, dim, query_data, N_QUERY, K, gt_indices.data());

    // 2. 创建索引
    std::cout << ">> [Step 2] Creating CAGRA index..." << std::endl;
    cagra::CagraIndex index(dim);

    // 3. 添加数据
    std::cout << ">> [Step 3] Adding " << N_BUILD << " vectors..." << std::endl;
    index.add(N_BUILD, base_data);

    // 4. Build (全量构建)
    std::cout << ">> [Step 4] Building index (this may take a while)..." << std::endl;
    try {
        index.build();
        std::cout << ">> [Step 4] Index build completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during index build: " << e.what() << std::endl;
        throw;
    }

    // 5. 执行查询
    std::cout << ">> Querying " << N_QUERY << " queries..." << std::endl;
    std::vector<int64_t> out_indices(N_QUERY * K);
    std::vector<float> out_dists(N_QUERY * K);

    auto t1 = std::chrono::high_resolution_clock::now();
    index.query(query_data, N_QUERY, K, out_indices.data(), out_dists.data());
    auto t2 = std::chrono::high_resolution_clock::now();

    double query_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 6. 统计结果
    double recall = calculate_recall(N_QUERY, K, gt_indices.data(), out_indices.data());
    double qps = N_QUERY * 1000.0 / query_time;

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Dataset Size: " << N_BUILD << " vectors" << std::endl;
    std::cout << "Query Count:  " << N_QUERY << " queries" << std::endl;
    std::cout << "Top-K:        " << K << std::endl;
    std::cout << "Query Time:   " << std::fixed << std::setprecision(3) << query_time << " ms" << std::endl;
    std::cout << "QPS:          " << std::setprecision(0) << (size_t)qps << std::endl;
    std::cout << "Recall@10:    " << std::setprecision(2) << recall * 100 << "%" << std::endl;

    // 7. 保存索引（可选）
    if (!index_path.empty()) {
        std::cout << ">> Saving index to " << index_path << "..." << std::endl;
        index.save(index_path);
    }
}

// 测试: 增量构建性能
void test_incremental_build(const float* base_data, int dim, int n_total) {
    std::cout << "\n[TEST] Incremental Build Performance" << std::endl;

    const int K = 10;
    const int N_QUERY = 100;

    // 1. 创建索引
    cagra::CagraIndex index(dim);

    // 2. 准备测试查询
    std::uniform_int_distribution<int> query_dist(0, n_total - 1);
    std::mt19937 rng(42);
    std::vector<float> queries(N_QUERY * dim);

    for (int i = 0; i < N_QUERY; ++i) {
        int query_idx = query_dist(rng);
        std::copy(&base_data[query_idx * dim],
                 &base_data[(query_idx + 1) * dim],
                 queries.data() + i * dim);
    }

    // 3. 测试不同数据量下的性能 (使用更大的规模)
    std::vector<int> test_sizes = {100000, 250000, 500000, static_cast<int>(n_total)};
    std::unique_ptr<cagra::CagraIndex> current_index;

    for (int test_size : test_sizes) {
        if (test_size > n_total) break;

        std::cout << "\n>>> Testing with " << test_size << " vectors:" << std::endl;

        // 重置索引
        current_index = std::make_unique<cagra::CagraIndex>(dim);
        current_index->add(test_size, base_data);
        current_index->build();

        // 生成GT
        std::vector<int64_t> gt_indices(N_QUERY * K);
        generate_ground_truth(base_data, test_size, dim, queries.data(), N_QUERY, K, gt_indices.data());

        // 查询测试
        std::vector<int64_t> out_indices(N_QUERY * K);
        std::vector<float> out_dists(N_QUERY * K);

        auto t1 = std::chrono::high_resolution_clock::now();
        current_index->query(queries.data(), N_QUERY, K, out_indices.data(), out_dists.data());
        auto t2 = std::chrono::high_resolution_clock::now();

        double query_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double recall = calculate_recall(N_QUERY, K, gt_indices.data(), out_indices.data());
        double qps = N_QUERY * 1000.0 / query_time;

        std::cout << "  Size: " << std::setw(8) << test_size
                  << " | Query: " << std::setw(6) << query_time << " ms"
                  << " | QPS: " << std::setw(8) << (size_t)qps
                  << " | Recall: " << std::setw(6) << recall * 100 << "%" << std::endl;
    }
}

// =============================================================================
// 主函数
// =============================================================================

int main() {
    try {
        // 配置CUDA设备
        int cuda_device = 0;
        cudaSetDevice(cuda_device);
        int current_device;
        cudaGetDevice(&current_device);
        std::cout << "Current CUDA device: " << current_device << std::endl;

        std::cout << "==========================================================" << std::endl;
        std::cout << "Original CAGRA HDFS Dataset Test (Baseline)" << std::endl;
        std::cout << "==========================================================" << std::endl;

        // 加载数据
        std::string base_path = "../data/HDFS.log-1M.fvecs";
        std::string query_path = "../data/HDFS.log-1M.query.fvecs";

        int dim, num_base, num_query;
        std::vector<float> base_data = load_fvecs(base_path, dim, num_base);
        std::vector<float> query_data = load_fvecs(query_path, dim, num_query);

        // 使用全部数据
        std::cout << "Using full dataset: " << num_base << " base vectors, " << num_query << " queries" << std::endl;

        std::cout << "\nDataset: " << num_base << " base vectors, " << num_query << " queries" << std::endl;
        std::cout << "Dimension: " << dim << std::endl;

        // 运行测试
        std::string index_path = "cagra_hdfs_index.bin";

        // 测试1: 标准构建和查询
        test_build_query(base_data.data(), dim, num_base, query_data.data(), num_query, index_path);

              std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Original CAGRA baseline test completed!" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}