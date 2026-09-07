#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(error));                \
        }                                                                       \
    } while (0)

struct Args {
    std::string data_path;
    std::string dataset;
    std::string csv_path;
    size_t num_vectors = 1'000'000;
    size_t num_buckets = 100;
    int batch_size = 1000;
    int rounds = 5;
    int topk = 10;
    int device = 0;
    uint32_t seed = 20'260'717;
    std::vector<double> ratios = {0.01, 0.10, 0.20, 1.0};
};

struct FvecsData {
    size_t rows;
    int dim;
    std::vector<float> values;
};

std::vector<double> parse_ratios(const std::string& value) {
    std::vector<double> ratios;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        double ratio = std::stod(token);
        if (ratio <= 0.0 || ratio > 1.0) {
            throw std::invalid_argument("ratios must be in (0, 1]");
        }
        ratios.push_back(ratio);
    }
    if (ratios.empty()) throw std::invalid_argument("at least one ratio is required");
    return ratios;
}

Args parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::invalid_argument(
            "Usage: gpu_bruteforce_baseline DATA.fvecs DATASET OUTPUT.csv "
            "[--n N] [--buckets B] [--batch Q] [--rounds R] [--k K] "
            "[--device D] [--seed S] [--ratios r1,r2,...]");
    }

    Args args;
    args.data_path = argv[1];
    args.dataset = argv[2];
    args.csv_path = argv[3];
    for (int i = 4; i < argc; ++i) {
        std::string key = argv[i];
        auto require_value = [&](const char* option) -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + option);
            }
            return argv[i];
        };
        if (key == "--n") args.num_vectors = std::stoull(require_value("--n"));
        else if (key == "--buckets") args.num_buckets = std::stoull(require_value("--buckets"));
        else if (key == "--batch") args.batch_size = std::stoi(require_value("--batch"));
        else if (key == "--rounds") args.rounds = std::stoi(require_value("--rounds"));
        else if (key == "--k") args.topk = std::stoi(require_value("--k"));
        else if (key == "--device") args.device = std::stoi(require_value("--device"));
        else if (key == "--seed") args.seed = std::stoul(require_value("--seed"));
        else if (key == "--ratios") args.ratios = parse_ratios(require_value("--ratios"));
        else throw std::invalid_argument("Unknown option: " + key);
    }

    if (args.num_vectors == 0 || args.num_buckets == 0 || args.batch_size <= 0 ||
        args.rounds <= 0 || args.topk <= 0) {
        throw std::invalid_argument("n, buckets, batch, rounds, and k must be positive");
    }
    if (args.num_buckets > args.num_vectors) {
        throw std::invalid_argument("bucket count cannot exceed vector count");
    }
    return args;
}

FvecsData load_fvecs_prefix(const std::string& path, size_t requested_rows) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open fvecs file: " + path);

    int dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || dim <= 0) throw std::runtime_error("Invalid fvecs dimension");
    input.seekg(0, std::ios::end);
    const size_t file_bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int) + static_cast<size_t>(dim) * sizeof(float);
    if (file_bytes % record_bytes != 0) {
        throw std::runtime_error("fvecs file size is not a whole number of records");
    }
    const size_t file_rows = file_bytes / record_bytes;
    if (file_rows < requested_rows) {
        throw std::runtime_error("fvecs file contains fewer rows than requested");
    }

    FvecsData data{requested_rows, dim, std::vector<float>(requested_rows * dim)};
    input.seekg(0, std::ios::beg);
    for (size_t row = 0; row < requested_rows; ++row) {
        int row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (row_dim != dim) {
            throw std::runtime_error("Inconsistent fvecs dimension at row " +
                                     std::to_string(row));
        }
        input.read(reinterpret_cast<char*>(data.values.data() + row * dim),
                   static_cast<size_t>(dim) * sizeof(float));
        if (!input) throw std::runtime_error("Unexpected end of fvecs file");
    }
    return data;
}

double milliseconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

std::string join_values(const std::vector<double>& values) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) output << ';';
        output << values[i];
    }
    return output.str();
}

void append_csv(const Args& args,
                int dim,
                double ratio,
                double avg_candidates,
                double avg_build_ms,
                double avg_search_ms,
                double avg_qps,
                double avg_qps_with_rebuild,
                const std::vector<double>& run_qps) {
    std::filesystem::path output_path(args.csv_path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    const bool write_header = !std::filesystem::exists(output_path) ||
                              std::filesystem::file_size(output_path) == 0;
    std::ofstream output(output_path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot open output CSV: " + args.csv_path);
    if (write_header) {
        output << "method,dataset,n,dim,buckets,range_ratio,range_pct,batch,k,rounds,"
                  "avg_candidates,avg_index_build_ms,avg_search_ms,avg_qps,"
                  "avg_qps_with_rebuild,run_qps\n";
    }
    output << std::fixed << std::setprecision(6)
           << "gpu_flat_l2," << args.dataset << ',' << args.num_vectors << ',' << dim
           << ',' << args.num_buckets << ',' << ratio << ',' << ratio * 100.0 << ','
           << args.batch_size << ',' << args.topk << ',' << args.rounds << ','
           << avg_candidates << ',' << avg_build_ms << ',' << avg_search_ms << ','
           << avg_qps << ',' << avg_qps_with_rebuild << ",\""
           << join_values(run_qps) << "\"\n";
}

void run(const Args& args, const FvecsData& data) {
    CUDA_CHECK(cudaSetDevice(args.device));
    faiss::gpu::StandardGpuResources resources;
    resources.setTempMemory(256ULL * 1024 * 1024);

    std::vector<float> queries(static_cast<size_t>(args.batch_size) * data.dim);
    std::vector<float> distances(static_cast<size_t>(args.batch_size) * args.topk);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(args.batch_size) * args.topk);
    const size_t bucket_size =
        (args.num_vectors + args.num_buckets - 1) / args.num_buckets;

    std::cout << "GPU FlatL2 baseline: dataset=" << args.dataset
              << " n=" << args.num_vectors << " dim=" << data.dim
              << " buckets=" << args.num_buckets << " batch=" << args.batch_size
              << " rounds=" << args.rounds << " device=" << args.device << std::endl;

    for (double ratio : args.ratios) {
        size_t span = static_cast<size_t>(
            std::ceil(static_cast<double>(args.num_buckets) * ratio));
        span = std::max<size_t>(1, std::min(span, args.num_buckets));
        const size_t max_start_bucket = args.num_buckets - span;
        const uint32_t ratio_key = static_cast<uint32_t>(std::llround(ratio * 1'000'000.0));
        std::mt19937 range_rng(args.seed ^ (ratio_key * 97'531u));

        std::vector<double> run_qps;
        std::vector<double> run_build_ms;
        std::vector<double> run_search_ms;
        std::vector<double> run_qps_with_rebuild;
        std::vector<size_t> candidate_counts;
        run_qps.reserve(args.rounds);

        for (int round = 0; round < args.rounds; ++round) {
            const size_t start_bucket = max_start_bucket == 0
                                            ? 0
                                            : std::uniform_int_distribution<size_t>(
                                                  0, max_start_bucket)(range_rng);
            const size_t end_bucket = start_bucket + span;
            const size_t candidate_begin =
                std::min(start_bucket * bucket_size, args.num_vectors);
            const size_t candidate_end =
                std::min(end_bucket * bucket_size, args.num_vectors);
            const size_t candidate_count = candidate_end - candidate_begin;
            if (candidate_count < static_cast<size_t>(args.topk)) {
                throw std::runtime_error("Filtered candidate set is smaller than top-k");
            }
            candidate_counts.push_back(candidate_count);

            std::mt19937 query_rng(
                args.seed ^ ratio_key ^ (static_cast<uint32_t>(round + 1) * 2'654'435'761u));
            std::uniform_int_distribution<size_t> query_distribution(0, candidate_count - 1);
            for (int query = 0; query < args.batch_size; ++query) {
                size_t local_id = query_distribution(query_rng);
                const float* source = data.values.data() +
                                      (candidate_begin + local_id) * data.dim;
                std::copy(source,
                          source + data.dim,
                          queries.data() + static_cast<size_t>(query) * data.dim);
            }

            faiss::gpu::GpuIndexFlatConfig config;
            config.device = args.device;
            config.useFloat16 = false;
            const auto build_start = std::chrono::steady_clock::now();
            faiss::gpu::GpuIndexFlatL2 index(&resources, data.dim, config);
            index.add(candidate_count,
                      data.values.data() + candidate_begin * static_cast<size_t>(data.dim));
            CUDA_CHECK(cudaDeviceSynchronize());
            const double build_ms = milliseconds_since(build_start);

            index.search(args.batch_size,
                         queries.data(),
                         args.topk,
                         distances.data(),
                         labels.data());
            CUDA_CHECK(cudaDeviceSynchronize());

            const auto search_start = std::chrono::steady_clock::now();
            index.search(args.batch_size,
                         queries.data(),
                         args.topk,
                         distances.data(),
                         labels.data());
            CUDA_CHECK(cudaDeviceSynchronize());
            const double search_ms = milliseconds_since(search_start);
            const double qps = static_cast<double>(args.batch_size) * 1000.0 / search_ms;
            const double qps_with_rebuild =
                static_cast<double>(args.batch_size) * 1000.0 / (build_ms + search_ms);

            run_build_ms.push_back(build_ms);
            run_search_ms.push_back(search_ms);
            run_qps.push_back(qps);
            run_qps_with_rebuild.push_back(qps_with_rebuild);
            std::cout << "  ratio=" << std::setw(5) << ratio
                      << " round=" << (round + 1) << '/' << args.rounds
                      << " range=[" << start_bucket << ',' << end_bucket << ")"
                      << " candidates=" << candidate_count
                      << " build_ms=" << std::fixed << std::setprecision(3) << build_ms
                      << " search_ms=" << search_ms
                      << " qps=" << std::setprecision(2) << qps << std::endl;
        }

        auto mean = [](const auto& values) {
            return std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<double>(values.size());
        };
        const double avg_candidates =
            std::accumulate(candidate_counts.begin(), candidate_counts.end(), 0.0) /
            static_cast<double>(candidate_counts.size());
        const double avg_build_ms = mean(run_build_ms);
        const double avg_search_ms = mean(run_search_ms);
        const double avg_qps = mean(run_qps);
        const double avg_qps_with_rebuild = mean(run_qps_with_rebuild);
        append_csv(args,
                   data.dim,
                   ratio,
                   avg_candidates,
                   avg_build_ms,
                   avg_search_ms,
                   avg_qps,
                   avg_qps_with_rebuild,
                   run_qps);
        std::cout << "  AVG ratio=" << ratio << " qps=" << avg_qps
                  << " qps_with_rebuild=" << avg_qps_with_rebuild << std::endl;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        FvecsData data = load_fvecs_prefix(args.data_path, args.num_vectors);
        run(args, data);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpu_bruteforce_baseline: " << error.what() << std::endl;
        return 1;
    }
}