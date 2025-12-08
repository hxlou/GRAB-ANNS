#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <set>

// System headers
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// FAISS headers
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>

// -----------------------------------------------------------------------------
// Timer Helper
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
// Data Parsing Helper
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
// Main Benchmark
// -----------------------------------------------------------------------------
int main() {
    // 1. Configuration
    std::string meta_path = "../data/hotpotqa_fullwiki_train.meta.json";
    std::string bin_path  = "../data/hotpotqa_fullwiki_train.bin";

    const size_t BASE_SIZE = 50000;
    const size_t INSERT_SIZE = 50000;
    const size_t TOTAL_SIZE = BASE_SIZE + INSERT_SIZE;
    
    // HNSW Params
    const int M = 32;
    const int efConstruction = 128;
    const int efSearch = 64;

    // Search Params
    const int SEARCH_BATCH_SIZE = 32;
    const int SEARCH_ITERATIONS = 10;
    const int TOTAL_QUERIES = SEARCH_BATCH_SIZE * SEARCH_ITERATIONS;
    const int K = 10; // Top-K

    std::cout << "==========================================================" << std::endl;
    std::cout << "FAISS HNSW (CPU) Benchmark" << std::endl;
    std::cout << "Base Size: " << BASE_SIZE << ", Insert Size: " << INSERT_SIZE << std::endl;
    std::cout << "Search: " << SEARCH_ITERATIONS << " iters x " << SEARCH_BATCH_SIZE << " queries" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 2. Load Data
    int dim = -1, file_total = -1;
    if (!parseMeta(meta_path, dim, file_total)) {
        std::cerr << "Error parsing meta file: " << meta_path << std::endl;
        return 1;
    }

    if (file_total < TOTAL_SIZE) {
        std::cerr << "Error: Not enough data. File has " << file_total 
                  << ", need " << TOTAL_SIZE << std::endl;
        return 1;
    }

    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) { std::cerr << "Error opening bin file." << std::endl; return 1; }
    
    size_t file_bytes = (size_t)file_total * dim * sizeof(float);
    const float* host_data = (const float*)mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) { std::cerr << "mmap failed." << std::endl; close(fd); return 1; }

    std::cout << ">> Data loaded via mmap. Dim: " << dim << std::endl;

    // 3. Initialize HNSW Index
    faiss::IndexHNSWFlat index(dim, M, faiss::METRIC_L2);
    index.hnsw.efConstruction = efConstruction;
    index.hnsw.efSearch = efSearch;

    // 4. Build Base Index (500k)
    std::cout << ">> [Step 1] Building Base Index (" << BASE_SIZE << " vectors)..." << std::endl;
    Timer timer;
    index.add(BASE_SIZE, host_data);
    std::cout << "   Base Build Time: " << timer.elapsed_ms() << " ms" << std::endl;

    // 5. Insert Benchmark (200k)
    std::cout << ">> [Step 2] Inserting " << INSERT_SIZE << " vectors..." << std::endl;
    const float* insert_data_ptr = host_data + BASE_SIZE * dim;
    
    timer.reset();
    index.add(INSERT_SIZE, insert_data_ptr);
    double insert_time_ms = timer.elapsed_ms();
    
    double ips = (INSERT_SIZE * 1000.0) / insert_time_ms;
    std::cout << "   Insert Time: " << insert_time_ms << " ms" << std::endl;
    std::cout << "   IPS: " << std::fixed << std::setprecision(2) << ips << " vectors/sec" << std::endl;

    // 6. Search Benchmark & Ground Truth Verification
    std::cout << ">> [Step 3] Preparing Search & Ground Truth..." << std::endl;

    // Create a Ground Truth Index (Brute Force) containing ALL data (700k)
    // Note: For very large datasets, this might be slow, but for 700k it's manageable on CPU
    faiss::IndexFlatL2 gt_index(dim);
    gt_index.add(TOTAL_SIZE, host_data); 

    // Generate random query indices
    std::vector<int> query_indices(TOTAL_QUERIES);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, TOTAL_SIZE - 1);
    for(int i=0; i<TOTAL_QUERIES; ++i) query_indices[i] = dis(gen);

    // Prepare query buffer
    std::vector<float> query_vectors(TOTAL_QUERIES * dim);
    for(int i=0; i<TOTAL_QUERIES; ++i) {
        const float* src = host_data + (size_t)query_indices[i] * dim;
        std::copy(src, src + dim, query_vectors.begin() + i * dim);
    }

    // Run Benchmark
    std::cout << "   Running " << SEARCH_ITERATIONS << " iterations (Batch=" << SEARCH_BATCH_SIZE << ")..." << std::endl;
    
    double total_search_time_ms = 0;
    long total_recall_hits = 0;

    for (int it = 0; it < SEARCH_ITERATIONS; ++it) {
        printf("search iteration %d / %d\r", it + 1, SEARCH_ITERATIONS);
        // Pointers for current batch
        const float* batch_queries = query_vectors.data() + it * SEARCH_BATCH_SIZE * dim;
        
        // Output buffers for HNSW
        std::vector<faiss::idx_t> I_hnsw(SEARCH_BATCH_SIZE * K);
        std::vector<float> D_hnsw(SEARCH_BATCH_SIZE * K);

        // A. Measure HNSW Search Time
        auto t1 = std::chrono::high_resolution_clock::now();
        index.search(SEARCH_BATCH_SIZE, batch_queries, K, D_hnsw.data(), I_hnsw.data());
        auto t2 = std::chrono::high_resolution_clock::now();
        total_search_time_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // B. Calculate Ground Truth (Brute Force)
        std::vector<faiss::idx_t> I_gt(SEARCH_BATCH_SIZE * K);
        std::vector<float> D_gt(SEARCH_BATCH_SIZE * K);
        gt_index.search(SEARCH_BATCH_SIZE, batch_queries, K, D_gt.data(), I_gt.data());

        // C. Calculate Recall@K
        for (int q = 0; q < SEARCH_BATCH_SIZE; ++q) {
            std::set<faiss::idx_t> gt_set;
            for (int k = 0; k < K; ++k) gt_set.insert(I_gt[q * K + k]);

            for (int k = 0; k < K; ++k) {
                if (gt_set.count(I_hnsw[q * K + k])) {
                    total_recall_hits++;
                }
            }
        }
    }

    double avg_recall = (double)total_recall_hits / (TOTAL_QUERIES * K);
    double qps = (TOTAL_QUERIES * 1000.0) / total_search_time_ms;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Final Results:" << std::endl;
    std::cout << "  IPS (Insertion): " << ips << " vec/sec" << std::endl;
    std::cout << "  QPS (Search):    " << qps << " queries/sec" << std::endl;
    std::cout << "  Recall@" << K << ":      " << (avg_recall * 100.0) << " %" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    // Cleanup
    munmap((void*)host_data, file_bytes);
    close(fd);

    return 0;
}