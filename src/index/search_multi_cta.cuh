#pragma once

#include "cagra.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace cagra {

void search_multi_cta_opt(const float* d_dataset,
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
                          uint32_t num_seeds_per_query,
                          uint32_t num_cta_per_query = 0,
                          cudaStream_t stream = 0);

void search_multi_cta_range_opt(const float* d_dataset,
                                uint32_t dim,
                                size_t num_dataset,
                                const uint32_t* d_graph,
                                const uint64_t* d_timestamps,
                                uint32_t graph_degree,
                                const float* d_queries,
                                int64_t num_queries,
                                int64_t k,
                                uint64_t start_bucket,
                                uint64_t end_bucket,
                                SearchParams params,
                                int64_t* d_out_indices,
                                float* d_out_dists,
                                const uint32_t* d_seeds,
                                uint32_t num_seeds_per_query,
                                uint32_t num_cta_per_query = 0,
                                cudaStream_t stream = 0);

uint32_t resolve_multi_cta_count(int64_t k, SearchParams params, uint32_t num_cta_per_query = 0);

uint32_t resolve_multi_cta_local_topk();

size_t multi_cta_intermediate_count(int64_t num_queries, uint32_t num_cta_per_query);

size_t multi_cta_hash_count(int64_t num_queries, uint32_t num_cta_per_query, uint32_t hash_bitlen);

size_t multi_cta_traversed_hash_count(int64_t num_queries, uint32_t hash_bitlen);

void search_multi_cta_opt_preallocated(const float* d_dataset,
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
                                       uint32_t* d_intermediate_indices,
                                       float* d_intermediate_dists,
                                       uint32_t* d_pre_hashmap,
                                       uint32_t* d_traversed_hashmap,
                                       const uint32_t* d_seeds,
                                       uint32_t num_seeds_per_query,
                                       uint32_t num_cta_per_query = 0,
                                       uint64_t rand_xor_mask = 0x9e3779b97f4a7c15ULL,
                                       const uint64_t* d_timestamps = nullptr,
                                       uint64_t start_bucket = 0,
                                       uint64_t end_bucket = 0,
                                       cudaStream_t stream = 0,
                                       float* profile_ms = nullptr,
                                       bool run_merge = true,
                                       uint64_t* d_stage_profile = nullptr);

void merge_multi_cta_results(uint32_t* d_intermediate_indices,
                             const float* d_intermediate_dists,
                             int64_t* d_out_indices,
                             float* d_out_dists,
                             int64_t num_queries,
                             int64_t k,
                             uint32_t num_cta_per_query,
                             uint32_t local_topk,
                             cudaStream_t stream = 0,
                             float* profile_ms = nullptr);

} // namespace cagra
