#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace cagra {

void build_time_partitioned_graph(const float* d_dataset,
                                  size_t total_num,
                                  uint32_t dim,
                                  uint32_t* d_graph,
                                  uint64_t* d_ts,
                                  uint64_t* h_ts,
                                  const std::vector<size_t>& bucket_sizes,
                                  uint32_t total_degree,
                                  uint32_t local_degree);

void search_opt(const float* d_dataset,
            uint32_t dim,
            size_t num_dataset,
            const uint32_t* d_graph,    
            uint32_t graph_degree,      
            const float* d_queries,
            int64_t num_queries,
            int64_t k,
            SearchParams params,
            int64_t* d_out_indices, 
            float* d_out_dists,
            const uint32_t* d_seeds,
            const uint32_t num_seeds_per_query,
            cudaStream_t stream = 0);

void search_bucket_opt(const float* d_dataset,
                       uint32_t dim,
                       size_t num_dataset,
                       const uint32_t* d_graph,
                       uint32_t total_degree,  // stride (32)
                       uint32_t local_degree,  // active (28)
                       const float* d_queries,
                       int64_t num_queries,
                       int64_t k,
                       SearchParams params,
                       int64_t* d_out_indices, 
                       float* d_out_dists,
                       const uint32_t* d_seeds,
                       const uint32_t num_seeds_per_query,
                       cudaStream_t stream = 0);

void search_bucket_range(const float* d_dataset,
                       uint32_t dim,
                       size_t num_dataset,
                       const uint32_t* d_graph,
                       uint64_t* d_ts,
                       uint32_t total_degree,  // stride (32)
                       uint32_t local_degree,  // active (28)
                       const float* d_queries,
                       int64_t num_queries,
                       int64_t k,
                       uint64_t start_bucket,
                       uint64_t end_bucket,
                       SearchParams params,
                       int64_t* d_out_indices, 
                       float* d_out_dists,
                       const uint32_t* d_seeds,
                       const uint32_t num_seeds_per_query);

void insert(const float* d_dataset,
                     uint32_t* d_graph,
                     const uint64_t* d_ts,
                     const uint32_t* d_seeds,
                     size_t num_existing,
                     size_t num_new,
                     bool use_heuristic,
                     int target_ts,
                     const float* d_queries,
                     uint32_t dim,
                     uint32_t total_degree,
                     uint32_t local_degree,
                     SearchParams params,
                     uint32_t num_seeds_per_query);

} // namespace cagra