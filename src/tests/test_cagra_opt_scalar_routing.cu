#include "../cagraIndexOpt.cuh"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_bucket(const cagra::CagraIndexOpt& index,
                   uint64_t scalar,
                   uint64_t expected_bucket) {
    uint64_t actual_bucket = index.route_scalar_to_bucket(scalar);
    if (actual_bucket != expected_bucket) {
        throw std::runtime_error(
            "scalar " + std::to_string(scalar) + " routed to bucket " +
            std::to_string(actual_bucket) + ", expected " +
            std::to_string(expected_bucket));
    }
}

void write_empty_v1_index(const char* path, uint32_t dim, uint32_t graph_degree,
                          size_t local_degree) {
    std::ofstream output(path, std::ios::binary);
    const char magic[4] = {'M', 'C', 'A', 'G'};
    const uint32_t version = 1;
    const size_t zero = 0;
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    output.write(reinterpret_cast<const char*>(&graph_degree), sizeof(graph_degree));
    output.write(reinterpret_cast<const char*>(&local_degree), sizeof(local_degree));
    output.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); // current_size
    output.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); // timestamps
    output.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); // buckets
    output.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); // data
    output.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); // graph
}

} // namespace

int main() {
    cudaError_t status = cudaSetDevice(0);
    if (status != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }

    constexpr uint32_t dim = 96;
    constexpr size_t count = 7;
    constexpr size_t vmm_bytes = 64ULL * 1024 * 1024;
    std::vector<float> vectors(count * dim, 0.0f);
    const uint64_t buckets[count] = {0, 0, 1, 1, 1, 2, 2};
    const uint64_t scalars[count] = {10, 20, 30, 35, 40, 40, 50};

    cagra::CagraIndexOpt index(dim, 4, 2, vmm_bytes);
    index.add(count, vectors.data(), buckets, scalars);
    index.build_scalar_to_bucket_map();

    if (index.scalar_to_bucket_map_size() != 6) {
        throw std::runtime_error("unexpected scalar-to-bucket map size");
    }

    expect_bucket(index, 5, 0);
    expect_bucket(index, 10, 0);
    expect_bucket(index, 25, 0);
    expect_bucket(index, 35, 1);
    expect_bucket(index, 37, 1);
    expect_bucket(index, 40, 2);
    expect_bucket(index, 45, 2);
    expect_bucket(index, 60, 2);

    const char* path = "/tmp/cagra_scalar_routing_test.idx";
    index.save(path);

    cagra::CagraIndexOpt loaded(dim, 4, 2, vmm_bytes);
    loaded.load(path);
    std::remove(path);

    if (loaded.scalar_to_bucket_map_size() != 6) {
        throw std::runtime_error("serialized scalar-to-bucket map size changed");
    }
    expect_bucket(loaded, 25, 0);
    expect_bucket(loaded, 40, 2);
    expect_bucket(loaded, 60, 2);

    const char* v1_path = "/tmp/cagra_scalar_routing_v1_test.idx";
    write_empty_v1_index(v1_path, dim, 4, 2);
    cagra::CagraIndexOpt loaded_v1(dim, 4, 2, vmm_bytes);
    loaded_v1.load(v1_path);
    std::remove(v1_path);
    if (loaded_v1.scalar_to_bucket_map_size() != 0) {
        throw std::runtime_error("v1 index unexpectedly contains a scalar map");
    }
    bool missing_map_rejected = false;
    try {
        loaded_v1.route_scalar_to_bucket(10);
    } catch (const std::logic_error&) {
        missing_map_rejected = true;
    }
    if (!missing_map_rejected) {
        throw std::runtime_error("v1 index routed a scalar without a scalar map");
    }

    std::cout << "PASSED: scalar routing and v1/v2 serialization" << std::endl;
    return 0;
}