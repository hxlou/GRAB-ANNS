#include "../src/cagraIndexOpt.cuh"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <map>
#include <numeric>

// Standard library
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <faiss/IndexFlat.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

// =============================================================================
// 1. Configuration
// =============================================================================

struct BuildConfig {
    size_t total_data_size;   // Total number of vectors
    size_t num_buckets;       // Number of buckets
    uint32_t graph_degree;    // Graph degree
    float base_ratio;         // Fraction assigned to the base index (0.0 to 1.0)

    std::string to_string() const {
        return "N=" + std::to_string(total_data_size) +
               " Bkt=" + std::to_string(num_buckets) +
               " Deg=" + std::to_string(graph_degree) +
               " Ratio=" + std::to_string(base_ratio);
    }
};

struct SearchConfig {
    uint32_t itopk;
    uint32_t width;
    uint32_t iter;

    std::string to_string() const {
        return "[k=" + std::to_string(itopk) +
               " w=" + std::to_string(width) +
               " i=" + std::to_string(iter) + "]";
    }
};

struct ResultStats {
    double insert_time_ms;
    double ips;             // Insertions Per Second
    double ratio;           // Query range ratio
    double avg_recall;
    double avg_qps;
    long long bound_errors;
    long long local_bound_errors;
};

// =============================================================================
// 2. Utilities and fvecs loader
// =============================================================================
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

void load_fvecs(const std::string& filename, std::vector<float>& data, int& dim, size_t& num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        exit(1);
    }
    in.read((char*)&dim, sizeof(int));
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t row_size = sizeof(int) + dim * sizeof(float);
    num = file_size / row_size;

    std::cout << ">> Loading SIFT from " << filename << " (Dim=" << dim << ", N=" << num << ")..." << std::endl;
    data.resize(num * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        int d;
        in.read((char*)&d, sizeof(int));
        if (d != dim) { std::cerr << "Dim mismatch!" << std::endl; exit(1); }
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
}

// Compute recall from ground-truth and search-result ID sets.
double calc_recall(size_t nq, int k, const int64_t* gt, const int64_t* res) {
    size_t correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j) gt_set.insert(gt[i * k + j]);
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(res[i * k + j])) correct++;
        }
    }
    return 100.0 * (double)correct / (nq * k);
}

void log_csv(const std::string& filename, const BuildConfig& b, const SearchConfig& s, const ResultStats& r) {
    std::ofstream file(filename, std::ios::app);
    if (file.tellp() == 0) {
        file << "Dataset,Buckets,Degree,BaseRatio,InsertTime(ms),IPS,Itopk,Width,Iter,RangeRatio,Recall,QPS,Errors\n";
    }
    file << b.total_data_size << "," << b.num_buckets << "," << b.graph_degree << "," << b.base_ratio << ","
         << r.insert_time_ms << "," << r.ips << ","
         << s.itopk << "," << s.width << "," << s.iter << ","
         << r.ratio << "," << r.avg_recall << "," << r.avg_qps << "," << r.bound_errors << "\n";
}

// =============================================================================
// 3. Evaluation workflow
// =============================================================================
void run_insert_benchmark_suite(
    const float* host_full_data,
    int dim,
    const BuildConfig& b_conf,
    const std::vector<SearchConfig>& search_configs,
    const std::vector<double>& range_ratios,
    const std::string& csv_file
) {
    std::cout << "\n>>> [CONFIG] " << b_conf.to_string() << std::endl;

    // -------------------------------------------------------------
    // 1. Split vectors between base construction and insertion.
    // -------------------------------------------------------------
    size_t bucket_size = b_conf.total_data_size / b_conf.num_buckets;
    if (bucket_size == 0) bucket_size = 1;

    // Scalar values associated with the source vectors
    std::vector<uint64_t> timestamps(b_conf.total_data_size);

    // Per-bucket base and insertion buffers
    std::vector<float> build_data;
    std::vector<uint64_t> build_ts;

    // Store insertion vectors separately for each bucket.
    std::vector<std::vector<float>> insert_data(b_conf.num_buckets);
    std::vector<std::vector<uint64_t>> insert_ts(b_conf.num_buckets);

    size_t n_build_total = (size_t)(b_conf.total_data_size * b_conf.base_ratio);
    build_data.reserve(n_build_total * dim);
    for (auto& vec : insert_data) {
        vec.reserve(((size_t)(bucket_size * (1.0 - b_conf.base_ratio)) + 1) * dim);
    }

    size_t split_point = (size_t)(bucket_size * b_conf.base_ratio);

    for(size_t i=0; i<b_conf.total_data_size; ++i) {
        uint64_t ts = i / bucket_size;
        timestamps[i] = ts;
        const float* vec = host_full_data + i * dim;

        // Assign the first base_ratio fraction of each bucket to the base index.
        if ((i % bucket_size) < split_point) {
            build_data.insert(build_data.end(), vec, vec + dim);
            build_ts.push_back(ts);
        } else {
            insert_data[ts].insert(insert_data[ts].end(), vec, vec + dim);
            insert_ts[ts].push_back(ts);
        }
    }

    std::cout << "   [Data] Base: " << build_ts.size() << ", Insert: " << insert_ts.size() << std::endl;

    // -------------------------------------------------------------
    // 2. Build & Insert
    // -------------------------------------------------------------
    cagra::CagraIndexOpt index(dim, b_conf.graph_degree, b_conf.graph_degree / 2);
    index.setBuildParams(b_conf.graph_degree * 2, b_conf.graph_degree);

    // Build
    Timer build_timer;
    index.add(build_ts.size(), build_data.data(), build_ts.data());
    index.build();
    double t_build = build_timer.elapsed_ms();
    std::cout << "   [Build] Time: " << t_build << " ms" << std::endl;

    // Insert
    // Use fixed insertion-search parameters for every evaluated dataset.
    index.setQueryParams(256, 4, 0, 50, 14);

    Timer insert_timer;
    // index.insert(insert_ts.size(), insert_data.data(), insert_ts.data());
    for (size_t bkt = 0; bkt < b_conf.num_buckets; ++bkt) {
        if (insert_ts[bkt].empty()) continue;
        if (bkt % 10 == 0) printf("====================== now is in bucket %lu ======================\n", bkt);
        index.insert(insert_ts[bkt].size(), insert_data[bkt].data(), insert_ts[bkt].data());
        // sleep(5);
    }
    // Read graph metadata from the constructed index.
    auto graph = index.get_graph();
    for (int i = 1145; i < 1000000; i+= 30000) {
        std::cout << "Node " << i << " neighbors: ";
        for (int j = 0; j < b_conf.graph_degree; ++j) {
            std::cout << graph[i * b_conf.graph_degree + j] << " ";
            if (j == (b_conf.graph_degree / 2 - 1)) std::cout << "  |  ";
        }
        std::cout << std::endl;
    }

    double t_insert = insert_timer.elapsed_ms();
    double ips = b_conf.total_data_size * (1.0 - b_conf.base_ratio) * 1000.0 / t_insert;

    std::cout << "   [Insert] Time: " << t_insert << " ms, IPS: " << (int)ips << std::endl;


    // -------------------------------------------------------------
    // 3. Evaluate search configurations and range ratios.
    // -------------------------------------------------------------
    const int NUM_ROUNDS = 5;
    const int QUERIES_PER_ROUND = 100;
    const int K = 10;
    std::mt19937 rng(12345);

    // Access the complete vector array stored by the index.
    const float* index_data_ptr = index.get_data();

    for (const auto& s_conf : search_configs) {
        index.setQueryParams(s_conf.itopk, s_conf.width, 0, s_conf.iter, 14);

        std::cout << "   [Search Config] " << s_conf.to_string() << std::endl;

        for (double ratio : range_ratios) {
            ResultStats stats = {}; // Zero init
            stats.insert_time_ms = t_insert;
            stats.ips = ips;
            stats.ratio = ratio;

            // Determine how many buckets are covered by the range.
            int span_buckets = (int)(b_conf.num_buckets * ratio);
            if (span_buckets < 1) span_buckets = 1;
            if (span_buckets > b_conf.num_buckets) span_buckets = b_conf.num_buckets;

            // Average metrics over multiple rounds.
            for (int round = 0; round < NUM_ROUNDS; ++round) {
                // A. Generate a range [start_bucket, end_bucket).
                int max_start = b_conf.num_buckets - span_buckets;
                int start_bucket = std::uniform_int_distribution<int>(0, max_start)(rng);
                int end_bucket = start_bucket + span_buckets;

                // =========================================================
                // B. Collect global IDs within the selected range.
                // =========================================================
                std::vector<uint32_t> range_global_ids;
                range_global_ids.reserve(span_buckets * (b_conf.total_data_size / b_conf.num_buckets) * 1.1);

                for (uint64_t ts = start_bucket; ts < end_bucket; ++ts) {
                    // Read bucket IDs through the index interface.
                    std::vector<uint32_t> bucket_ids = index.get_ids_by_timestamp(ts);
                    range_global_ids.insert(range_global_ids.end(), bucket_ids.begin(), bucket_ids.end());
                }

                size_t range_data_size = range_global_ids.size();
                if (range_data_size == 0) continue;

                // C. Gather vectors into a contiguous temporary array.
                std::vector<float> range_vectors(range_data_size * dim);
                for (size_t i = 0; i < range_data_size; ++i) {
                    uint32_t gid = range_global_ids[i];
                    const float* src = index_data_ptr + (size_t)gid * dim;
                    std::copy(src, src + dim, range_vectors.data() + i * dim);
                }

                // D. Sample queries from the vectors in the range.
                std::vector<float> queries(QUERIES_PER_ROUND * dim);
                std::uniform_int_distribution<size_t> q_dist(0, range_data_size - 1);

                for (int i = 0; i < QUERIES_PER_ROUND; ++i) {
                    size_t local_idx = q_dist(rng);
                    std::copy(range_vectors.data() + local_idx * dim,
                              range_vectors.data() + (local_idx + 1) * dim,
                              queries.data() + i * dim);
                }

                // E. Compute ground truth with a CPU FAISS index over range_vectors.
                // FAISS returns offsets in [0, range_data_size).
                std::vector<int64_t> gt_local_indices(QUERIES_PER_ROUND * K);
                std::vector<float> gt_dists(QUERIES_PER_ROUND * K);
                {
                    faiss::IndexFlatL2 cpu_index(dim);
                    cpu_index.add(range_data_size, range_vectors.data());
                    cpu_index.search(QUERIES_PER_ROUND, queries.data(), K, gt_dists.data(), gt_local_indices.data());
                }

                // F. Map local FAISS offsets to global vector IDs.
                std::vector<int64_t> gt_global_indices(QUERIES_PER_ROUND * K);
                for (size_t i = 0; i < gt_global_indices.size(); ++i) {
                    gt_global_indices[i] = (int64_t)range_global_ids[gt_local_indices[i]];
                }

                // =========================================================
                // G. Execute the GRAB-ANNS search.
                // =========================================================
                std::vector<int64_t> out_indices(QUERIES_PER_ROUND * K);
                std::vector<float> out_dists(QUERIES_PER_ROUND * K);

                Timer t;
                // Use the configured active degree, including remote edges.
                if (ratio < 0.02) {
                    index.query_local(queries.data(),
                                    QUERIES_PER_ROUND,
                                    K,
                                    (uint64_t)start_bucket,
                                    out_indices.data(),
                                    out_dists.data(),
                                    b_conf.graph_degree);
                } else if (ratio > 0.99) {
                    index.query(queries.data(),
                                    QUERIES_PER_ROUND,
                                    K,
                                    0,
                                    UINT64_MAX,
                                    out_indices.data(),
                                    out_dists.data());
                } else {
                    index.query_range(queries.data(),
                                    QUERIES_PER_ROUND,
                                    K,
                                    (uint64_t)start_bucket,
                                    (uint64_t)end_bucket,
                                    out_indices.data(),
                                    out_dists.data(),
                                    b_conf.graph_degree);
                }


                double ms = t.elapsed_ms();

                // H. Aggregate metrics and validate result ranges.
                std::unordered_set<uint32_t> valid_gids_set(range_global_ids.begin(), range_global_ids.end());
                for (int i = 0; i < QUERIES_PER_ROUND * K; ++i) {
                    int64_t gid = out_indices[i];
                    if (gid != -1) {
                        // A returned ID outside valid_ids violates the range predicate.
                        if (valid_gids_set.find((uint32_t)gid) == valid_gids_set.end()) {
                            stats.local_bound_errors++;
                        }
                    }
                }

                stats.avg_recall += calc_recall(QUERIES_PER_ROUND, K, gt_global_indices.data(), out_indices.data());
                stats.avg_qps += QUERIES_PER_ROUND * 1000.0 / ms;
            }

            stats.avg_recall /= NUM_ROUNDS;
            stats.avg_qps /= NUM_ROUNDS;

            std::cout << "      Ratio=" << (int)(ratio*100) << "% | Recall="
                      << std::fixed << std::setprecision(2) << stats.avg_recall << "% | QPS="
                      << (int)stats.avg_qps << " | IPS=" << (int)stats.ips << " | Errors=" << stats.local_bound_errors << std::endl;

            log_csv(csv_file, b_conf, s_conf, stats);
        }
    }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0]
                  << " SIFT_BASE.fvecs [output.csv]" << std::endl;
        return 2;
    }
    printf("Using cuda device %d\n", CUDA_DEVICE_ID);
    CHECK_CUDA(cudaSetDevice(CUDA_DEVICE_ID));

    std::string sift_path = argv[1];
    std::string csv_file =
        (argc == 3) ? argv[2] : "results/exp4_ingestion/grab_insert.csv";

    // 1. Load the configured dataset.
    std::vector<float> host_full_data;
    int dim = 0;
    size_t file_total = 0;
    load_fvecs(sift_path, host_full_data, dim, file_total);

    // Dataset dimensionality is supplied by the selected configuration.
    if (dim != 128) {
        std::cerr << "Warning: SIFT dim is 128. Ensure config.cuh DIM is 128!" << std::endl;
    }

    // ==========================================================
    // Evaluation parameters
    // ==========================================================

    // A. Build Configs
    std::vector<BuildConfig> build_configs = {
        // DataSize, Buckets, Degree, BaseRatio
        {1000000, 100, 32, 0.5f},
        {1000000, 100, 32, 0.1f},
        {1000000, 100, 64, 0.5f},
        {1000000, 100, 64, 0.1f},
    };

    // B. Search Configs
    std::vector<SearchConfig> search_configs = {
        // Itopk, Width, Iter
        {128, 4, 50},
        {128, 4, 100},
        {128, 4, 150},
        {128, 4, 200},
        {128, 6, 50},
        {128, 6, 100},
        {128, 6, 150},
        {128, 6, 200},
        {256, 4, 50},
        {256, 4, 100},
        {256, 4, 150},
        {256, 4, 200},
        {256, 6, 50},
        {256, 6, 100},
        {256, 6, 150},
        {256, 6, 200},
        {512, 4, 50},
        {512, 4, 100},
        {512, 4, 150},
        {512, 4, 200},
    };

    // C. Range Ratios
    std::vector<double> range_ratios = {0.01, 0.1, 0.2, 1.0}; // 0.01 is approximately one bucket; 1.0 covers all buckets.

    std::cout << "Starting Accurate SIFT Insert+Range Benchmark..." << std::endl;

    for (const auto& b_conf : build_configs) {
        if (b_conf.total_data_size > file_total) continue;
        run_insert_benchmark_suite(host_full_data.data(), dim, b_conf, search_configs, range_ratios, csv_file);
    }

    return 0;
}
