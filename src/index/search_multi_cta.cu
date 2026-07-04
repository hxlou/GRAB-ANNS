#include "search_multi_cta.cuh"

#include "bitonic.cuh"
#include "compute_distance.cuh"
#include "config.cuh"
#include "hashmap.cuh"
#include "search.cuh"
#include "smem_cal.cuh"
#include "warp_merge_sort.cuh"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace cagra {
namespace device {

__device__ __forceinline__ void pickup_next_parents(uint32_t* terminate_flag,
                                                    uint32_t* next_parent_indices,
                                                    uint32_t* internal_topk_indices,
                                                    uint32_t internal_topk_size,
                                                    uint32_t search_width)
{
    uint32_t lane_id = threadIdx.x % 32;
    constexpr uint32_t MSB_MASK = 0x80000000;
    constexpr uint32_t INVALID_IDX = 0xFFFFFFFF;

    for (uint32_t i = lane_id; i < search_width; i += 32) {
        next_parent_indices[i] = INVALID_IDX;
    }

    uint32_t num_new_parents = 0;
    for (uint32_t j = threadIdx.x; j < internal_topk_size; j += 32) {
        uint32_t node_id = internal_topk_indices[j];
        int is_new_parent = ((node_id & MSB_MASK) == 0 && node_id != INVALID_IDX);
        uint32_t vote_mask = __ballot_sync(0xffffffff, is_new_parent);

        if (is_new_parent) {
            uint32_t rank_in_warp = __popc(vote_mask & ((1u << threadIdx.x) - 1));
            uint32_t global_rank = num_new_parents + rank_in_warp;
            if (global_rank < search_width) {
                next_parent_indices[global_rank] = node_id;
                internal_topk_indices[j] |= MSB_MASK;
            }
        }

        num_new_parents += __popc(vote_mask);
        if (num_new_parents >= search_width) break;
    }

    if (lane_id == 0) {
        *terminate_flag = (num_new_parents == 0) ? 1 : 0;
    }
}

__device__ __forceinline__ void pickup_next_parent_multi_cta(uint32_t* terminate_flag,
                                                             uint32_t* next_parent_indices,
                                                             uint32_t* internal_topk_indices,
                                                             float* internal_topk_dists,
                                                             uint32_t internal_topk_size,
                                                             uint32_t* traversed_hash,
                                                             uint32_t traversed_hash_bitlen)
{
    constexpr uint32_t MSB_MASK = 0x80000000;
    constexpr uint32_t INVALID_IDX = 0xFFFFFFFF;

    if (threadIdx.x == 0) {
        next_parent_indices[0] = INVALID_IDX;
        *terminate_flag = 1;
        for (uint32_t i = 0; i < internal_topk_size; ++i) {
            uint32_t node_id = internal_topk_indices[i];
            if (node_id == INVALID_IDX || (node_id & MSB_MASK)) continue;

            uint32_t clean_id = node_id & ~MSB_MASK;
            bool usable = traversed_hash == nullptr ||
                          cagra::hashmap::insert(traversed_hash, traversed_hash_bitlen, clean_id);
            if (usable) {
                next_parent_indices[0] = clean_id;
                internal_topk_indices[i] = clean_id | MSB_MASK;
                *terminate_flag = 0;
                break;
            }

            internal_topk_indices[i] = INVALID_IDX;
            internal_topk_dists[i] = FLT_MAX;
        }
    }
}

__device__ inline float compute_l2_for_dim(const float* query_buffer,
                                           const float* node_ptr,
                                           uint32_t dim)
{
    if (dim == 1024) return cagra::device::calc_l2_dist_1024(query_buffer, node_ptr);
    if (dim == 2048) return cagra::device::calc_l2_dist_2048(query_buffer, node_ptr);
    if (dim == 960) return cagra::device::calc_l2_dist_960(query_buffer, node_ptr);
    if (dim == 256) return cagra::device::calc_l2_dist_256(query_buffer, node_ptr);
    if (dim == 128) return cagra::device::calc_l2_dist_128(query_buffer, node_ptr);
    if (dim == 96) return cagra::device::calc_l2_dist_96(query_buffer, node_ptr);
    if (threadIdx.x == 0) printf("[multi_cta] unsupported dimension %u\n", dim);
    return FLT_MAX;
}

__device__ inline void init_multi_cta_nodes(uint32_t* result_indices,
                                           float* result_dists,
                                           const float* query_buffer,
                                           const float* dataset_ptr,
                                           size_t num_dataset,
                                           uint32_t dim,
                                           uint32_t queue_capacity,
                                           uint32_t num_pickup,
                                           const uint32_t* seed_ptr,
                                           uint32_t num_seeds_per_query,
                                           uint64_t rand_xor_mask,
                                           uint32_t* visited_hash,
                                           uint32_t visited_hash_bitlen,
                                           uint32_t* traversed_hash,
                                           uint32_t traversed_hash_bitlen,
                                           uint32_t block_id,
                                           uint32_t num_blocks)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32;

    for (uint32_t i = tid; i < queue_capacity; i += blockDim.x) {
        result_indices[i] = 0xFFFFFFFF;
        result_dists[i] = FLT_MAX;
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < num_pickup; i += num_warps) {
        uint32_t gid = block_id + num_blocks * i;
        uint32_t node_id = 0xFFFFFFFF;
        if (seed_ptr != nullptr && gid < num_seeds_per_query) {
            node_id = seed_ptr[gid];
        }
        if (node_id >= num_dataset) {
            node_id = (rand_xor_mask * static_cast<uint64_t>(gid + 1)) % num_dataset;
        }

        bool usable = true;
        if (lane_id == 0) {
            usable = cagra::hashmap::insert(visited_hash, visited_hash_bitlen, node_id);
            if (usable && traversed_hash != nullptr) {
                usable = !cagra::hashmap::search(traversed_hash, traversed_hash_bitlen, node_id);
            }
        }
        usable = __shfl_sync(0xffffffff, usable, 0);

        float dist = FLT_MAX;
        if (usable) {
            const float* node_ptr = dataset_ptr + static_cast<size_t>(node_id) * dim;
            dist = compute_l2_for_dim(query_buffer, node_ptr, dim);
        }

        if (lane_id == 0) {
            result_indices[i] = usable ? node_id : 0xFFFFFFFF;
            result_dists[i] = usable ? dist : FLT_MAX;
        }
    }
    __syncthreads();
}

__device__ inline void compute_distance_to_child_nodes_multi_cta(
    uint32_t* candidate_indices,
    float* candidate_dists,
    const float* query_buffer,
    const float* dataset_ptr,
    const uint32_t* knn_graph,
    const uint64_t* timestamps_ptr,
    uint32_t graph_degree,
    uint32_t dim,
    uint32_t* visited_hash,
    uint32_t visited_hash_bitlen,
    uint32_t* traversed_hash,
    uint32_t traversed_hash_bitlen,
    const uint32_t* parent_list,
    uint32_t search_width,
    uint64_t start_bucket,
    uint64_t end_bucket)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t num_warps = blockDim.x / 32;
    const uint32_t total_tasks = search_width * graph_degree;

    for (uint32_t task_id = warp_id; task_id < total_tasks; task_id += num_warps) {
        uint32_t parent_idx = task_id / graph_degree;
        uint32_t neighbor_offset = task_id % graph_degree;
        uint32_t neighbor_id = 0xFFFFFFFF;
        bool usable = false;

        uint32_t parent_id = parent_list[parent_idx];
        if (parent_id != 0xFFFFFFFF) {
            neighbor_id = knn_graph[static_cast<size_t>(parent_id) * graph_degree + neighbor_offset];
            usable = neighbor_id != 0xFFFFFFFF;
        }

        if (usable && timestamps_ptr != nullptr) {
            uint64_t bucket_id = __ldg(&timestamps_ptr[neighbor_id]);
            usable = bucket_id >= start_bucket && bucket_id < end_bucket;
        }

        if (lane_id == 0 && usable) {
            usable = cagra::hashmap::insert(visited_hash, visited_hash_bitlen, neighbor_id);
            if (usable && traversed_hash != nullptr) {
                usable = !cagra::hashmap::search(traversed_hash, traversed_hash_bitlen, neighbor_id);
            }
        }
        usable = __shfl_sync(0xffffffff, usable, 0);

        float dist = FLT_MAX;
        if (usable) {
            const float* node_ptr = dataset_ptr + static_cast<size_t>(neighbor_id) * dim;
            dist = compute_l2_for_dim(query_buffer, node_ptr, dim);
        }

        if (lane_id == 0) {
            candidate_indices[task_id] = usable ? neighbor_id : 0xFFFFFFFF;
            candidate_dists[task_id] = usable ? dist : FLT_MAX;
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(1024, 1) void search_multi_cta_kernel(
    uint32_t* intermediate_indices,       // [num_queries, num_cta, local_topk]
    float* intermediate_dists,            // [num_queries, num_cta, local_topk]
    const float* queries_ptr,             // [num_queries, dim]
    const float* dataset_ptr,             // [N, dim]
    const uint32_t* knn_graph,            // [N, graph_degree]
    const uint64_t* timestamps_ptr,        // [N], optional for range filter
    const uint32_t* seed_ptr,             // [num_queries, num_seeds_per_query]
    uint32_t num_seeds_per_query,
    uint32_t num_queries,
    size_t num_dataset,
    uint32_t dim,
    uint32_t graph_degree,
    uint32_t local_topk,
    uint32_t search_width,
    uint32_t max_iterations,
    uint32_t num_seeds,
    uint64_t rand_xor_mask,
    uint32_t hash_bitlen,
    uint64_t start_bucket,
    uint64_t end_bucket,
    uint32_t* pre_hashmap,
    uint32_t* traversed_hashmap,
    uint32_t traversed_hash_bitlen,
    uint32_t queue_capacity)
{
    extern __shared__ uint8_t smem[];

    size_t offset = 0;
    float* query_buffer = reinterpret_cast<float*>(smem + offset);
    offset += (dim * sizeof(float) + 15) & ~15;

    uint32_t* visited_hash = nullptr;
    uint32_t linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    if (hash_bitlen < 14) {
        visited_hash = reinterpret_cast<uint32_t*>(smem + offset);
        offset += (((1u << hash_bitlen) * sizeof(uint32_t)) + 15) & ~15;
    } else {
        visited_hash = pre_hashmap + static_cast<size_t>(linear_block_id) * (1u << hash_bitlen);
    }

    uint32_t* result_indices = reinterpret_cast<uint32_t*>(smem + offset);
    offset += (queue_capacity * sizeof(uint32_t) + 15) & ~15;

    float* result_dists = reinterpret_cast<float*>(smem + offset);
    offset += (queue_capacity * sizeof(float) + 15) & ~15;

    uint32_t* parent_list = reinterpret_cast<uint32_t*>(smem + offset);
    offset += (search_width * sizeof(uint32_t) + 15) & ~15;

    volatile uint32_t* terminate_flag = reinterpret_cast<uint32_t*>(smem + offset);

    const uint32_t query_id = blockIdx.x;
    const uint32_t cta_id = blockIdx.y;
    const uint32_t num_cta = gridDim.y;
    const uint32_t tid = threadIdx.x;

    if (query_id >= num_queries) return;

    const float* global_query = queries_ptr + static_cast<size_t>(query_id) * dim;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = global_query[i];
    }

    if (tid == 0) *terminate_flag = 0;
    cagra::hashmap::init(visited_hash, hash_bitlen);
    __syncthreads();

    uint32_t* query_traversed_hash = traversed_hashmap == nullptr
        ? nullptr
        : traversed_hashmap + static_cast<size_t>(query_id) * (1u << traversed_hash_bitlen);

    uint64_t local_rand_mask =
        rand_xor_mask ^ (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(cta_id + 1));

    const uint32_t* local_seed_ptr = seed_ptr == nullptr
        ? nullptr
        : seed_ptr + static_cast<size_t>(query_id) * num_seeds_per_query;
    uint32_t block_id = cta_id + num_cta * query_id;
    uint32_t num_blocks = num_cta * num_queries;
    init_multi_cta_nodes(
        result_indices,
        result_dists,
        query_buffer,
        dataset_ptr,
        num_dataset,
        dim,
        queue_capacity,
        num_seeds,
        local_seed_ptr,
        num_seeds_per_query,
        local_rand_mask,
        visited_hash,
        hash_bitlen,
        query_traversed_hash,
        traversed_hash_bitlen,
        block_id,
        num_blocks);
    __syncthreads();

    for (uint32_t iter = 0; iter < max_iterations; ++iter) {
        if (queue_capacity == 64 && tid < 32) {
            cagra::merge::load_sort_store<2>(result_dists, result_indices, 64);
        } else if (queue_capacity == 128 && tid < 32) {
            cagra::merge::load_sort_store<4>(result_dists, result_indices, 128);
        } else if (queue_capacity == 256 && tid < 32) {
            cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
        } else if (queue_capacity == 512 && tid < 32) {
            cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
        } else if (tid == 0) {
            printf("[multi_cta] unsupported queue_capacity=%u\n", queue_capacity);
        }
        __syncthreads();

        if (tid < 32) {
            cagra::device::pickup_next_parent_multi_cta(
                const_cast<uint32_t*>(terminate_flag),
                parent_list,
                result_indices,
                result_dists,
                local_topk,
                query_traversed_hash,
                traversed_hash_bitlen);
        }
        __syncthreads();

        if (*terminate_flag == 1) break;

        cagra::hashmap::init(visited_hash, hash_bitlen);
        __syncthreads();
        for (uint32_t i = tid; i < queue_capacity; i += blockDim.x) {
            uint32_t idx = result_indices[i];
            if (idx == 0xFFFFFFFF) continue;
            cagra::hashmap::insert(visited_hash, hash_bitlen, idx & 0x7FFFFFFF);
        }
        __syncthreads();

        cagra::device::compute_distance_to_child_nodes_multi_cta(
            result_indices + local_topk,
            result_dists + local_topk,
            query_buffer,
            dataset_ptr,
            knn_graph,
            timestamps_ptr,
            graph_degree,
            dim,
            visited_hash,
            hash_bitlen,
            query_traversed_hash,
            traversed_hash_bitlen,
            parent_list,
            search_width,
            start_bucket,
            end_bucket);
        __syncthreads();
    }

    if (queue_capacity == 64 && tid < 32) {
        cagra::merge::load_sort_store<2>(result_dists, result_indices, 64);
    } else if (queue_capacity == 128 && tid < 32) {
        cagra::merge::load_sort_store<4>(result_dists, result_indices, 128);
    } else if (queue_capacity == 256 && tid < 32) {
        cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
    } else if (queue_capacity == 512 && tid < 32) {
        cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
    }
    __syncthreads();

    size_t out_base =
        (static_cast<size_t>(query_id) * num_cta + cta_id) * static_cast<size_t>(local_topk);
    for (uint32_t i = tid; i < local_topk; i += blockDim.x) {
        uint32_t idx = result_indices[i];
        bool valid = idx != 0xFFFFFFFF;
        idx &= 0x7FFFFFFF;
        if (valid && query_traversed_hash != nullptr && (result_indices[i] & 0x80000000) == 0) {
            valid = cagra::hashmap::insert(query_traversed_hash, traversed_hash_bitlen, idx);
        }
        intermediate_indices[out_base + i] = valid ? idx : 0xFFFFFFFF;
        intermediate_dists[out_base + i] = valid ? result_dists[i] : FLT_MAX;
    }
}

__global__ void merge_multi_cta_results_kernel(
    const uint32_t* intermediate_indices,
    const float* intermediate_dists,
    int64_t* out_indices,
    float* out_dists,
    uint32_t num_queries,
    uint32_t num_cta,
    uint32_t local_topk,
    uint32_t topk)
{
    uint32_t query_id = blockIdx.x;
    if (query_id >= num_queries) return;

    if (threadIdx.x == 0) {
        size_t base = static_cast<size_t>(query_id) * num_cta * local_topk;
        for (uint32_t out_k = 0; out_k < topk; ++out_k) {
            float best_dist = FLT_MAX;
            uint32_t best_pos = 0xFFFFFFFF;
            for (uint32_t i = 0; i < num_cta * local_topk; ++i) {
                uint32_t idx = intermediate_indices[base + i];
                float dist = intermediate_dists[base + i];
                if (idx == 0xFFFFFFFF) continue;

                bool duplicate = false;
                for (uint32_t prev = 0; prev < out_k; ++prev) {
                    if (out_indices[static_cast<size_t>(query_id) * topk + prev] ==
                        static_cast<int64_t>(idx)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate && dist < best_dist) {
                    best_dist = dist;
                    best_pos = i;
                }
            }

            size_t dst = static_cast<size_t>(query_id) * topk + out_k;
            if (best_pos == 0xFFFFFFFF) {
                out_indices[dst] = -1;
                out_dists[dst] = FLT_MAX;
            } else {
                out_indices[dst] = static_cast<int64_t>(intermediate_indices[base + best_pos]);
                out_dists[dst] = best_dist;
                const_cast<uint32_t*>(intermediate_indices)[base + best_pos] = 0xFFFFFFFF;
            }
        }
    }
}

} // namespace device

uint32_t resolve_multi_cta_count(int64_t k, SearchParams params, uint32_t num_cta_per_query)
{
    constexpr uint32_t local_topk = 32;
    uint32_t topk = static_cast<uint32_t>(k);
    uint32_t global_itopk = std::max(topk, params.itopk_size);
    if (num_cta_per_query == 0) {
        num_cta_per_query = std::max(params.search_width, (global_itopk + local_topk - 1) / local_topk);
    }
    return std::max(1u, num_cta_per_query);
}

size_t multi_cta_intermediate_count(int64_t num_queries, uint32_t num_cta_per_query)
{
    constexpr uint32_t local_topk = 32;
    return static_cast<size_t>(num_queries) * num_cta_per_query * local_topk;
}

size_t multi_cta_hash_count(int64_t num_queries, uint32_t num_cta_per_query, uint32_t hash_bitlen)
{
    if (hash_bitlen <= 13) return 0;
    return static_cast<size_t>(num_queries) * num_cta_per_query * (size_t{1} << hash_bitlen);
}

size_t multi_cta_traversed_hash_count(int64_t num_queries, uint32_t hash_bitlen)
{
    return static_cast<size_t>(num_queries) * (size_t{1} << hash_bitlen);
}

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
                          uint32_t num_cta_per_query,
                          cudaStream_t stream)
{
    if (d_graph == nullptr) {
        throw std::runtime_error("Graph is null!");
    }

    num_cta_per_query = resolve_multi_cta_count(k, params, num_cta_per_query);

    uint32_t* d_intermediate_indices = nullptr;
    float* d_intermediate_dists = nullptr;
    size_t intermediate_count = multi_cta_intermediate_count(num_queries, num_cta_per_query);
    CUDA_CHECK(cudaMalloc(&d_intermediate_indices, intermediate_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_intermediate_dists, intermediate_count * sizeof(float)));

    uint32_t* d_pre_hashmap = nullptr;
    size_t hash_count = multi_cta_hash_count(num_queries, num_cta_per_query, params.hash_bitlen);
    if (hash_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_pre_hashmap, hash_count * sizeof(uint32_t)));
    }
    uint32_t* d_traversed_hashmap = nullptr;
    size_t traversed_hash_count = multi_cta_traversed_hash_count(num_queries, params.hash_bitlen);
    CUDA_CHECK(cudaMalloc(&d_traversed_hashmap, traversed_hash_count * sizeof(uint32_t)));

    std::random_device rd;
    uint64_t rand_xor_mask = (static_cast<uint64_t>(rd()) << 32) ^ rd();

    search_multi_cta_opt_preallocated(d_dataset,
                                      dim,
                                      num_dataset,
                                      d_graph,
                                      graph_degree,
                                      d_queries,
                                      num_queries,
                                      k,
                                      params,
                                      d_out_indices,
                                      d_out_dists,
                                      d_intermediate_indices,
                                      d_intermediate_dists,
                                      d_pre_hashmap,
                                      d_traversed_hashmap,
                                      d_seeds,
                                      num_seeds_per_query,
                                      num_cta_per_query,
                                      rand_xor_mask,
                                      nullptr,
                                      0,
                                      0,
                                      stream,
                                      nullptr,
                                      true);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_intermediate_indices));
    CUDA_CHECK(cudaFree(d_intermediate_dists));
    if (d_pre_hashmap != nullptr) CUDA_CHECK(cudaFree(d_pre_hashmap));
    CUDA_CHECK(cudaFree(d_traversed_hashmap));
}

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
                                uint32_t num_cta_per_query,
                                cudaStream_t stream)
{
    if (d_graph == nullptr || d_timestamps == nullptr) {
        throw std::runtime_error("Graph/timestamps are null!");
    }

    num_cta_per_query = resolve_multi_cta_count(k, params, num_cta_per_query);

    uint32_t* d_intermediate_indices = nullptr;
    float* d_intermediate_dists = nullptr;
    size_t intermediate_count = multi_cta_intermediate_count(num_queries, num_cta_per_query);
    CUDA_CHECK(cudaMalloc(&d_intermediate_indices, intermediate_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_intermediate_dists, intermediate_count * sizeof(float)));

    uint32_t* d_pre_hashmap = nullptr;
    size_t hash_count = multi_cta_hash_count(num_queries, num_cta_per_query, params.hash_bitlen);
    if (hash_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_pre_hashmap, hash_count * sizeof(uint32_t)));
    }
    uint32_t* d_traversed_hashmap = nullptr;
    size_t traversed_hash_count = multi_cta_traversed_hash_count(num_queries, params.hash_bitlen);
    CUDA_CHECK(cudaMalloc(&d_traversed_hashmap, traversed_hash_count * sizeof(uint32_t)));

    constexpr uint64_t rand_xor_mask = 0x9e3779b97f4a7c15ULL;
    search_multi_cta_opt_preallocated(d_dataset,
                                      dim,
                                      num_dataset,
                                      d_graph,
                                      graph_degree,
                                      d_queries,
                                      num_queries,
                                      k,
                                      params,
                                      d_out_indices,
                                      d_out_dists,
                                      d_intermediate_indices,
                                      d_intermediate_dists,
                                      d_pre_hashmap,
                                      d_traversed_hashmap,
                                      d_seeds,
                                      num_seeds_per_query,
                                      num_cta_per_query,
                                      rand_xor_mask,
                                      d_timestamps,
                                      start_bucket,
                                      end_bucket,
                                      stream,
                                      nullptr,
                                      true);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_intermediate_indices));
    CUDA_CHECK(cudaFree(d_intermediate_dists));
    if (d_pre_hashmap != nullptr) CUDA_CHECK(cudaFree(d_pre_hashmap));
    CUDA_CHECK(cudaFree(d_traversed_hashmap));
}

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
                                       uint32_t num_cta_per_query,
                                       uint64_t rand_xor_mask,
                                       const uint64_t* d_timestamps,
                                       uint64_t start_bucket,
                                       uint64_t end_bucket,
                                       cudaStream_t stream,
                                       float* profile_ms,
                                       bool run_merge)
{
    if (d_graph == nullptr) {
        throw std::runtime_error("Graph is null!");
    }

    uint32_t local_topk = 32;
    uint32_t local_search_width = 1;
    num_cta_per_query = resolve_multi_cta_count(k, params, num_cta_per_query);

    uint32_t raw_needed = local_topk + local_search_width * graph_degree;
    uint32_t queue_capacity = cagra::detail::next_power_of_2(raw_needed);
    queue_capacity = std::min(queue_capacity, 512u);
    if (queue_capacity < 64) queue_capacity = 64;

    auto align16 = [](size_t bytes) { return (bytes + 15) & ~size_t{15}; };
    size_t smem_size = 0;
    smem_size += align16(static_cast<size_t>(dim) * sizeof(float));
    if (params.hash_bitlen < 14) {
        smem_size += align16((size_t{1} << params.hash_bitlen) * sizeof(uint32_t));
    }
    smem_size += align16(static_cast<size_t>(queue_capacity) * sizeof(uint32_t));
    smem_size += align16(static_cast<size_t>(queue_capacity) * sizeof(float));
    smem_size += align16(static_cast<size_t>(local_search_width) * sizeof(uint32_t));
    smem_size += 16;

    size_t hash_count = multi_cta_hash_count(num_queries, num_cta_per_query, params.hash_bitlen);
    size_t traversed_hash_count = multi_cta_traversed_hash_count(num_queries, params.hash_bitlen);
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    if (profile_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }

    if (hash_count > 0) {
        if (profile_ms != nullptr) CUDA_CHECK(cudaEventRecord(ev_start, stream));
        CUDA_CHECK(cudaMemsetAsync(d_pre_hashmap, 0xFF, hash_count * sizeof(uint32_t), stream));
        if (profile_ms != nullptr) {
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            profile_ms[0] += ms;
        }
    }
    CUDA_CHECK(cudaMemsetAsync(d_traversed_hashmap, 0xFF,
                               traversed_hash_count * sizeof(uint32_t),
                               stream));

    uint32_t num_seeds = std::min(queue_capacity, std::max(graph_degree, num_seeds_per_query));

    uint32_t block_size = 1024;
    if (const char* block_env = std::getenv("CAGRA_MULTI_CTA_BLOCK_SIZE")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(block_env, nullptr, 10));
        if (requested == 64 || requested == 128 || requested == 256 ||
            requested == 512 || requested == 1024) {
            block_size = requested;
        }
    }

    dim3 grid(static_cast<uint32_t>(num_queries), num_cta_per_query);
    dim3 block(block_size);
    if (profile_ms != nullptr) CUDA_CHECK(cudaEventRecord(ev_start, stream));
    cagra::device::search_multi_cta_kernel<<<grid, block, smem_size, stream>>>(
        d_intermediate_indices,
        d_intermediate_dists,
        d_queries,
        d_dataset,
        d_graph,
        d_timestamps,
        d_seeds,
        num_seeds_per_query,
        static_cast<uint32_t>(num_queries),
        num_dataset,
        dim,
        graph_degree,
        local_topk,
        local_search_width,
        params.max_iterations,
        num_seeds,
        rand_xor_mask,
        params.hash_bitlen,
        start_bucket,
        end_bucket,
        d_pre_hashmap,
        d_traversed_hashmap,
        params.hash_bitlen,
        queue_capacity);
    CUDA_CHECK(cudaGetLastError());
    if (profile_ms != nullptr) {
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        profile_ms[1] += ms;
    }

    if (run_merge) {
        merge_multi_cta_results(d_intermediate_indices,
                                d_intermediate_dists,
                                d_out_indices,
                                d_out_dists,
                                num_queries,
                                k,
                                num_cta_per_query,
                                stream,
                                profile_ms);
    }

    if (profile_ms != nullptr) {
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
    }
}

void merge_multi_cta_results(uint32_t* d_intermediate_indices,
                             const float* d_intermediate_dists,
                             int64_t* d_out_indices,
                             float* d_out_dists,
                             int64_t num_queries,
                             int64_t k,
                             uint32_t num_cta_per_query,
                             cudaStream_t stream,
                             float* profile_ms)
{
    constexpr uint32_t local_topk = 32;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    if (profile_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
    }

    cagra::device::merge_multi_cta_results_kernel<<<static_cast<uint32_t>(num_queries), 1, 0, stream>>>(
        d_intermediate_indices,
        d_intermediate_dists,
        d_out_indices,
        d_out_dists,
        static_cast<uint32_t>(num_queries),
        num_cta_per_query,
        local_topk,
        static_cast<uint32_t>(k));
    CUDA_CHECK(cudaGetLastError());

    if (profile_ms != nullptr) {
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        profile_ms[2] += ms;
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
    }
}

} // namespace cagra
