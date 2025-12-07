#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>

// 系统库
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS 头文件 (HNSW 是 CPU 索引)
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/utils.h>

// -----------------------------------------------------------------------------
// 计时器
// -----------------------------------------------------------------------------
class Timer {
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// -----------------------------------------------------------------------------
// 数据加载辅助 (复用之前的逻辑)
// -----------------------------------------------------------------------------
bool parseMeta(const std::string& path, int& dim, int& total) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line, content;
    while (std::getline(file, line)) content += line;
    try {
        auto parse_val = [&](const std::string& key) -> int {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return -1;
            size_t start = content.find(":", pos) + 1;
            size_t end = content.find(",", start);
            if (end == std::string::npos) end = content.find("}", start);
            return std::stoi(content.substr(start, end - start));
        };
        dim = parse_val("dim");
        total = parse_val("total"); 
        if (total == -1) total = parse_val("count");
        return (dim > 0 && total > 0);
    } catch (...) { return false; }
}

// -----------------------------------------------------------------------------
// 主程序
// -----------------------------------------------------------------------------
int main() {
    // 1. 路径配置 (请确保文件存在)
    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    // 2. 实验参数
    const size_t BASE_SIZE = 50000;   // 初始构建大小
    const size_t INSERT_SIZE = 50000; // 插入大小
    const size_t TOTAL_SIZE = BASE_SIZE + INSERT_SIZE;
    
    // HNSW 参数
    const int M = 32;          // 每个节点的边数 (同 CAGRA 的 graph_degree)
    const int efConstruction = 128; // 构建时的搜索深度 (越大越准但越慢)

    std::cout << "==========================================================" << std::endl;
    std::cout << "FAISS HNSW (CPU) Benchmark" << std::endl;
    std::cout << "Base: " << BASE_SIZE << ", Insert: " << INSERT_SIZE << std::endl;
    std::cout << "HNSW Config: M=" << M << ", efConstruction=" << efConstruction << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 3. 加载数据
    int dim = -1, file_total = -1;
    if (!parseMeta(meta_path, dim, file_total)) {
        std::cerr << "Error parsing meta file." << std::endl;
        return 1;
    }
    
    if (file_total < TOTAL_SIZE) {
        std::cerr << "Error: File only has " << file_total << " vectors, need " << TOTAL_SIZE << std::endl;
        return 1;
    }

    int fd = open(bin_path.c_str(), O_RDONLY);
    size_t file_bytes = (size_t)file_total * dim * sizeof(float);
    // 使用 mmap 读取，避免拷贝
    const float* host_data = (const float*)mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) { std::cerr << "mmap failed." << std::endl; return 1; }

    std::cout << ">> Data loaded (Mapped). Dim: " << dim << std::endl;

    // 4. 初始化 FAISS HNSW 索引
    // IndexHNSWFlat: HNSW 图结构 + 原始向量存储 (Flat)
    faiss::IndexHNSWFlat index(dim, M, faiss::METRIC_L2);
    index.hnsw.efConstruction = efConstruction;
    
    // 也可以设置多线程构建
    // omp_set_num_threads(16); 

    // 5. 构建 Base 索引 (5w)
    std::cout << ">> [Step 1] Building Base Index (" << BASE_SIZE << ")..." << std::endl;
    Timer timer;
    
    // faiss::Index 的 add 方法会将数据拷贝到内部存储
    index.add(BASE_SIZE, host_data);
    
    double build_time = timer.elapsed_ms();
    std::cout << "   Base Build Time: " << build_time << " ms" << std::endl;

    // 6. 执行插入 (5w) 并统计耗时
    std::cout << ">> [Step 2] Inserting " << INSERT_SIZE << " vectors..." << std::endl;
    
    const float* insert_data_ptr = host_data + BASE_SIZE * dim;
    
    timer.reset();
    index.add(INSERT_SIZE, insert_data_ptr);
    double insert_time = timer.elapsed_ms();

    // 7. 统计结果
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Insertion Performance Report (CPU HNSW):" << std::endl;
    std::cout << "  Count:       " << INSERT_SIZE << std::endl;
    std::cout << "  Total Time:  " << insert_time << " ms" << std::endl;
    std::cout << "  Throughput:  " << (INSERT_SIZE * 1000.0 / insert_time) << " vec/sec" << std::endl;
    std::cout << "  Avg Latency: " << (insert_time / INSERT_SIZE) << " ms/vec" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    // 8. 简单验证 (Search Sanity Check)
    std::cout << ">> [Verification] Running simple search..." << std::endl;
    int k = 10;
    int num_queries = 100;
    std::vector<float> queries(num_queries * dim);
    std::vector<faiss::idx_t> I(num_queries * k);
    std::vector<float> D(num_queries * k);

    // 随机取 100 个做查询
    // (取自刚才插入的后半部分数据，确保能搜到新插入的内容)
    for(int i=0; i<num_queries; ++i) {
        size_t idx = BASE_SIZE + i; 
        std::copy(host_data + idx * dim, host_data + (idx + 1) * dim, queries.data() + i * dim);
    }

    index.hnsw.efSearch = 64; // 搜索时的参数
    
    timer.reset();
    index.search(num_queries, queries.data(), k, D.data(), I.data());
    double search_time = timer.elapsed_ms();

    // 检查 Recall@1 (Self-search 应该是自己)
    int recall = 0;
    for(int i=0; i<num_queries; ++i) {
        if (I[i*k] == (BASE_SIZE + i)) recall++;
    }

    std::cout << "   Search Time: " << search_time << " ms" << std::endl;
    std::cout << "   Recall@1: " << recall << "/" << num_queries << " (" << recall << "%)" << std::endl;

    // 清理
    munmap((void*)host_data, file_bytes);
    close(fd);

    return 0;
}