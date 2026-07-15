#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cagraIndexOpt.cuh"

namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
    std::exit(1);
  }
}

std::vector<float> load_fvecs(
    const std::string& path, int& dim, std::size_t& count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Cannot open dataset: " + path);
  }

  const std::size_t bytes = static_cast<std::size_t>(input.tellg());
  input.seekg(0);
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (dim <= 0) {
    throw std::runtime_error("Invalid fvecs dimension");
  }

  const std::size_t record_bytes = sizeof(int) + dim * sizeof(float);
  count = bytes / record_bytes;
  input.seekg(0);

  std::vector<float> vectors(count * dim);
  for (std::size_t i = 0; i < count; ++i) {
    int row_dim = 0;
    input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
    if (row_dim != dim) {
      throw std::runtime_error("Inconsistent fvecs dimension");
    }
    input.read(
        reinterpret_cast<char*>(vectors.data() + i * dim),
        dim * sizeof(float));
  }
  return vectors;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr
        << "Usage: " << argv[0]
        << " DATASET.fvecs DATASET_NAME [N=1000000] [M=32] [output.csv]"
        << std::endl;
    return 2;
  }

  check_cuda(cudaSetDevice(0));

  int dim = 0;
  std::size_t available = 0;
  std::vector<float> vectors = load_fvecs(argv[1], dim, available);

  const std::string dataset = argv[2];
  const std::size_t n =
      (argc >= 4) ? std::stoull(argv[3]) : 1000000;
  const std::uint32_t degree =
      (argc >= 5) ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 32;
  const std::string output =
      (argc >= 6) ? argv[5] : "results/exp4_ingestion/grab_build.csv";

  if (n > available) {
    std::cerr << "Requested " << n << " vectors, input has " << available
              << std::endl;
    return 2;
  }

  const std::size_t num_buckets = std::max<std::size_t>(1, n / 10000);
  const std::size_t bucket_size = (n + num_buckets - 1) / num_buckets;
  std::vector<std::uint64_t> timestamps(n);
  for (std::size_t i = 0; i < n; ++i) {
    timestamps[i] = std::min<std::size_t>(i / bucket_size, num_buckets - 1);
  }

  cagra::CagraIndexOpt index(dim, degree);
  index.setBuildParams(degree * 2, degree);

  const auto begin = std::chrono::steady_clock::now();
  index.add(n, vectors.data(), timestamps.data());
  index.build();
  check_cuda(cudaDeviceSynchronize());
  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - begin).count();

  std::ofstream csv(output, std::ios::app);
  if (csv.tellp() == 0) {
    csv << "method,dataset,M,K,K_Search,range_pct,recall,qps,comps,"
           "build_time,ips\n";
  }
  csv << "lightCagra," << dataset << "," << degree << ",400,,,,,,"
      << seconds << "," << (n / seconds) << "\n";

  std::cout << "GRAB build: dataset=" << dataset << " N=" << n
            << " M=" << degree << " time=" << seconds << "s" << std::endl;
  return 0;
}
