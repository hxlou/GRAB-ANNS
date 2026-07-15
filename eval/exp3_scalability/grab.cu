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

// FAISS CPU API
#include <faiss/IndexFlat.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

// =============================================================================
// 1. Configuration
// =============================================================================
struct BuildConfig {
    size_t total_data_size;   // Number of vectors
    size_t num_buckets;       // Number of buckets
    uint32_t graph_degree;    // Graph degree

    std::string to_string() const {
        return "N=" + std::to_string(total_data_size) +
               " Bkt=" + std::to_string(num_buckets) +
               " Deg=" + std::to_string(graph_degree);
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

struct RangeStats {
    double ratio = 0.0;
    double avg_recall = 0.0;
    double avg_qps = 0.0;
    long long bound_errors = 0;
};

// =============================================================================
// 2. Precomputed query task
// =============================================================================
const int NUM_ROUNDS = 5;
const int QUERIES_PER_ROUND = 32;
const int K = 10;

struct BenchmarkTask {
    int start_bucket;
    int end_bucket;
    size_t range_global_start;
    size_t range_global_end;

    std::vector<float> queries;      // [QUERIES_PER_ROUND * dim]
    std::vector<int64_t> gt_indices; // [QUERIES_PER_ROUND * K] (Global ID)
};

// =============================================================================
// 3. Utilities and fvecs loader
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

    std::cout << ">> Loading Data from " << filename << "..." << std::endl;
    std::cout << "   Dim: " << dim << ", Count: " << num << std::endl;

    data.resize(num * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        int d;
        in.read((char*)&d, sizeof(int));
        if (d != dim) exit(1);
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
}

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

void log_csv(const std::string& filename, const BuildConfig& b, const SearchConfig& s, double build_time, const RangeStats& r) {
    std::ofstream file(filename, std::ios::app);
    if (file.tellp() == 0) {
        file << "Dataset,Buckets,Degree,BuildTime(ms),Itopk,Width,Iter,RangeRatio,Recall,QPS,BoundErrors\n";
    }
    file << b.total_data_size << "," << b.num_buckets << "," << b.graph_degree << "," << build_time << ","
         << s.itopk << "," << s.width << "," << s.iter << ","
         << r.ratio << "," << r.avg_recall << "," << r.avg_qps << "," << r.bound_errors << "\n";
}

// =============================================================================
// 4. Task generation and ground-truth computation
// =============================================================================
std::map<double, std::vector<BenchmarkTask>> generate_tasks(
    const float* host_data,
    int dim,
    const BuildConfig& b_conf,
    const std::vector<double>& ratios
) {
    std::cout << "   [Pre-computation] Generating Queries and GT for all ratios..." << std::endl;
    Timer t;

    std::map<double, std::vector<BenchmarkTask>> tasks_map;
    std::mt19937 rng(12345);
    size_t bucket_size = b_conf.total_data_size / b_conf.num_buckets;

    for (double ratio : ratios) {
        std::vector<BenchmarkTask> tasks;
        tasks.reserve(NUM_ROUNDS);

        int span_buckets = (int)(b_conf.num_buckets * ratio);
        if (span_buckets < 1) span_buckets = 1;
        if (span_buckets > b_conf.num_buckets) span_buckets = b_conf.num_buckets;

        for (int round = 0; round < NUM_ROUNDS; ++round) {
            BenchmarkTask task;

            // A. Generate a range.
            int max_start = b_conf.num_buckets - span_buckets;
            task.start_bucket = std::uniform_int_distribution<int>(0, max_start)(rng);
            task.end_bucket = task.start_bucket + span_buckets;

            task.range_global_start = task.start_bucket * bucket_size;
            task.range_global_end = std::min((size_t)task.end_bucket * bucket_size, b_conf.total_data_size);
            size_t range_len = task.range_global_end - task.range_global_start;
            const float* range_data_ptr = host_data + task.range_global_start * dim;

            // B. Sample queries.
            task.queries.resize(QUERIES_PER_ROUND * dim);
            std::uniform_int_distribution<int> q_dist(0, range_len - 1);
            for (int i = 0; i < QUERIES_PER_ROUND; ++i) {
                int local_idx = q_dist(rng);
                std::copy(range_data_ptr + local_idx * dim,
                          range_data_ptr + (local_idx + 1) * dim,
                          task.queries.data() + i * dim);
            }

            // C. Compute ground truth once with FAISS.
            task.gt_indices.resize(QUERIES_PER_ROUND * K);
            std::vector<float> gt_dists(QUERIES_PER_ROUND * K);

            // Build a temporary FAISS index over the selected range.
            faiss::IndexFlatL2 cpu_index(dim);
            cpu_index.add(range_len, range_data_ptr);

            // Search the temporary index.
            std::vector<int64_t> local_indices(QUERIES_PER_ROUND * K);
            cpu_index.search(QUERIES_PER_ROUND, task.queries.data(), K, gt_dists.data(), local_indices.data());

            // Convert local result IDs to global IDs.
            for(int i=0; i<QUERIES_PER_ROUND * K; ++i) {
                task.gt_indices[i] = local_indices[i] + task.range_global_start;
            }

            tasks.push_back(std::move(task));
        }
        tasks_map[ratio] = std::move(tasks);
    }

    std::cout << "   [Pre-computation] Done in " << t.elapsed_ms() << " ms." << std::endl;
    return tasks_map;
}

// =============================================================================
// 5. Evaluation with precomputed tasks
// =============================================================================
void run_range_benchmark_fast(
    cagra::CagraIndexOpt& index,
    int dim,
    const BuildConfig& b_conf,
    const SearchConfig& s_conf,
    double build_time,
    const std::map<double, std::vector<BenchmarkTask>>& tasks_map,
    const std::string& csv_file
) {
    // Set search parameters.
    index.setQueryParams(s_conf.itopk, s_conf.width, 0, s_conf.iter, 14);

    // std::cout << "   [Search Config] " << s_conf.to_string() << std::endl;

    for (const auto& kv : tasks_map) {
        double ratio = kv.first;
        const auto& tasks = kv.second; // NUM_ROUNDS tasks for this range ratio

        RangeStats stats = {};
        stats.ratio = ratio;

        for (const auto& task : tasks) {
            std::vector<int64_t> out_indices(QUERIES_PER_ROUND * K);
            std::vector<float> out_dists(QUERIES_PER_ROUND * K);

            // D. CAGRA Range Search
            Timer t;

            // Use graph_degree as active_degree to include remote edges.
            if (ratio <= 0.90f) {
                index.query_range(task.queries.data(), QUERIES_PER_ROUND, K,
                                  (uint64_t)task.start_bucket, (uint64_t)task.end_bucket,
                                  out_indices.data(), out_dists.data(), b_conf.graph_degree);
            } else {
                index.query(task.queries.data(), QUERIES_PER_ROUND, K, 0, 0, out_indices.data(), out_dists.data(), nullptr, 0);
            }
            double ms = t.elapsed_ms();

            // E. Aggregate metrics and check range validity.
            for (int i = 0; i < QUERIES_PER_ROUND * K; ++i) {
                int64_t gid = out_indices[i];
                if (gid != -1) {
                    if (gid < task.range_global_start || gid >= task.range_global_end) {
                        stats.bound_errors++;
                    }
                }
            }

            stats.avg_recall += calc_recall(QUERIES_PER_ROUND, K, task.gt_indices.data(), out_indices.data());
            stats.avg_qps += QUERIES_PER_ROUND * 1000.0 / ms;
        }

        stats.avg_recall /= NUM_ROUNDS;
        stats.avg_qps /= NUM_ROUNDS;

        // Print one summary line per configuration.
        // std::cout << "      Ratio=" << (int)(ratio*100) << "% | Recall="
        //           << std::fixed << std::setprecision(2) << stats.avg_recall << "% | QPS="
        //           << (int)stats.avg_qps << std::endl;

        log_csv(csv_file, b_conf, s_conf, build_time, stats);
    }
}


// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " DEEP10M.fvecs [output.csv]" << std::endl;
        return 2;
    }

    int cuda_device = 0;
    CHECK_CUDA(cudaSetDevice(cuda_device));

    const std::string data_path = argv[1];
    const std::string csv_file =
        (argc == 3) ? argv[2] : "results/exp3_scalability/grab.csv";

    std::vector<float> host_full_data;
    int dim = 0;
    size_t file_total = 0;
    load_fvecs(data_path, host_full_data, dim, file_total);

    std::vector<BuildConfig> build_configs;
    for (size_t size = 1000000; size <= 10000000; size += 1000000) {
        BuildConfig config;
        config.total_data_size = size;
        config.num_buckets = size / 10000;
        config.graph_degree = 64;
        build_configs.push_back(config);
    }

    const std::vector<SearchConfig> search_configs = {
        {512, 4, 100},
    };
    const std::vector<double> test_ratios = {0.10};

    std::cout << "Starting GRAB scalability benchmark on " << data_path << std::endl;
    for (const auto& b_conf : build_configs) {
        if (b_conf.total_data_size > file_total) {
            std::cout << "Skipping N=" << b_conf.total_data_size
                      << " because the input contains only " << file_total
                      << " vectors" << std::endl;
            continue;
        }

        size_t bucket_size = b_conf.total_data_size / b_conf.num_buckets;
        std::vector<uint64_t> timestamps(b_conf.total_data_size);
        for (size_t i = 0; i < b_conf.total_data_size; ++i) {
            timestamps[i] = i / bucket_size;
        }

        cagra::CagraIndexOpt index(dim, b_conf.graph_degree);
        index.setBuildParams(b_conf.graph_degree * 2, b_conf.graph_degree);

        Timer timer;
        index.add(b_conf.total_data_size, host_full_data.data(), timestamps.data());
        index.build();
        double build_time = timer.elapsed_ms();

        auto tasks_map =
            generate_tasks(host_full_data.data(), dim, b_conf, test_ratios);
        for (const auto& s_conf : search_configs) {
            run_range_benchmark_fast(
                index, dim, b_conf, s_conf, build_time, tasks_map, csv_file);
        }
    }
    return 0;
}
