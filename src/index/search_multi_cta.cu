#include "search_multi_cta.cuh"

#include "bitonic.cuh"
#include "compute_distance.cuh"
#include "config.cuh"
#include "hashmap.cuh"
#include "multi_cta_sort.cuh"
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

__device__ __forceinline__ void sort_multi_cta_buffer(float* result_dists,
                                                      uint32_t* result_indices,
                                                      uint32_t queue_capacity,
                                                      bool use_bitonic_sorter)
{
    if (threadIdx.x >= 32) return;

    if (use_bitonic_sorter) {
        if (queue_capacity == 64) {
            cagra::multi_cta_sort::load_sort_store<2>(result_dists, result_indices, 64);
        } else if (queue_capacity == 128) {
            cagra::multi_cta_sort::load_sort_store<4>(result_dists, result_indices, 128);
        } else if (queue_capacity == 256) {
            cagra::multi_cta_sort::load_sort_store<8>(result_dists, result_indices, 256);
        } else if (queue_capacity == 512) {
            cagra::multi_cta_sort::load_sort_store<16>(result_dists, result_indices, 512);
        }
    } else {
        if (queue_capacity == 64) {
            cagra::merge::load_sort_store<2>(result_dists, result_indices, 64);
        } else if (queue_capacity == 128) {
            cagra::merge::load_sort_store<4>(result_dists, result_indices, 128);
        } else if (queue_capacity == 256) {
            cagra::merge::load_sort_store<8>(result_dists, result_indices, 256);
        } else if (queue_capacity == 512) {
            cagra::merge::load_sort_store<16>(result_dists, result_indices, 512);
        }
    }
}

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
                                                             uint32_t search_width,
                                                             uint32_t* traversed_hash,
                                                             uint32_t traversed_hash_bitlen,
                                                             bool support_traversed_remove)
{
    constexpr uint32_t MSB_MASK = 0x80000000;
    constexpr uint32_t INVALID_IDX = 0xFFFFFFFF;

    if (threadIdx.x == 0) {
        *terminate_flag = 1;
        for (uint32_t i = 0; i < search_width; ++i) {
            next_parent_indices[i] = INVALID_IDX;
        }

        uint32_t num_parents = 0;
        for (uint32_t i = 0; i < internal_topk_size; ++i) {
            uint32_t node_id = internal_topk_indices[i];
            if (node_id == INVALID_IDX || (node_id & MSB_MASK)) continue;

            uint32_t clean_id = node_id & ~MSB_MASK;
            bool usable = traversed_hash == nullptr ||
                          (support_traversed_remove
                               ? cagra::hashmap::insert_support_remove(
                                     traversed_hash, traversed_hash_bitlen, clean_id)
                               : cagra::hashmap::insert(
                                     traversed_hash, traversed_hash_bitlen, clean_id));
            if (usable) {
                next_parent_indices[num_parents] = clean_id;
                internal_topk_indices[i] = clean_id | MSB_MASK;
                *terminate_flag = 0;
                ++num_parents;
                if (num_parents >= search_width) break;
                continue;
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

template <uint32_t TeamSize>
__device__ __forceinline__ float compute_l2_team(const float* query_buffer,
                                                 const float* node_ptr,
                                                 uint32_t dim)
{
    static_assert(TeamSize == 4 || TeamSize == 8 || TeamSize == 16 || TeamSize == 32,
                  "TeamSize must be a power-of-two sub-warp size");
    const uint32_t lane_in_warp = threadIdx.x & 31u;
    const uint32_t lane_in_team = lane_in_warp & (TeamSize - 1u);
    const uint32_t team_start = lane_in_warp & ~(TeamSize - 1u);
    const uint32_t team_mask = TeamSize == 32
        ? 0xffffffffu
        : (((1u << TeamSize) - 1u) << team_start);

    float sum_sq = 0.0f;
    for (uint32_t i = lane_in_team; i < dim; i += TeamSize) {
        float diff = query_buffer[i] - node_ptr[i];
        sum_sq += diff * diff;
    }

    #pragma unroll
    for (uint32_t offset = TeamSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(team_mask, sum_sq, offset);
    }
    return sum_sq;
}

template <uint32_t TeamSize, uint32_t Dim>
__device__ __forceinline__ float compute_l2_team_fixed(const float* query_buffer,
                                                       const float* node_ptr)
{
    static_assert(TeamSize == 4 || TeamSize == 8 || TeamSize == 16 || TeamSize == 32,
                  "TeamSize must be a power-of-two sub-warp size");
    const uint32_t lane_in_warp = threadIdx.x & 31u;
    const uint32_t lane_in_team = lane_in_warp & (TeamSize - 1u);
    const uint32_t team_start = lane_in_warp & ~(TeamSize - 1u);
    const uint32_t team_mask = TeamSize == 32
        ? 0xffffffffu
        : (((1u << TeamSize) - 1u) << team_start);

    float sum_sq = 0.0f;
    #pragma unroll
    for (uint32_t i = lane_in_team; i < Dim; i += TeamSize) {
        float diff = query_buffer[i] - node_ptr[i];
        sum_sq += diff * diff;
    }

    #pragma unroll
    for (uint32_t offset = TeamSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(team_mask, sum_sq, offset);
    }
    return sum_sq;
}

__device__ __forceinline__ float compute_l2_for_team(const float* query_buffer,
                                                     const float* node_ptr,
                                                     uint32_t dim,
                                                     uint32_t team_size)
{
    if (dim == 96 && team_size == 8) {
        return compute_l2_team_fixed<8, 96>(query_buffer, node_ptr);
    }
    if (dim == 128 && team_size == 8) {
        return compute_l2_team_fixed<8, 128>(query_buffer, node_ptr);
    }
    if (dim == 256 && team_size == 16) {
        return compute_l2_team_fixed<16, 256>(query_buffer, node_ptr);
    }
    if (team_size == 4) return compute_l2_team<4>(query_buffer, node_ptr, dim);
    if (team_size == 8) return compute_l2_team<8>(query_buffer, node_ptr, dim);
    if (team_size == 16) return compute_l2_team<16>(query_buffer, node_ptr, dim);
    return compute_l2_team<32>(query_buffer, node_ptr, dim);
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
                                           uint32_t num_random_samplings,
                                           uint64_t rand_xor_mask,
                                           uint32_t* visited_hash,
                                           uint32_t visited_hash_bitlen,
                                           uint32_t* traversed_hash,
                                           uint32_t traversed_hash_bitlen,
                                           uint32_t block_id,
                                           uint32_t num_blocks,
                                           bool support_traversed_remove,
                                           uint32_t distance_team_size)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_in_warp = tid & 31u;
    const uint32_t lane_in_team = lane_in_warp & (distance_team_size - 1u);
    const uint32_t team_start = lane_in_warp & ~(distance_team_size - 1u);
    const uint32_t team_id = tid / distance_team_size;
    const uint32_t num_teams = blockDim.x / distance_team_size;
    const uint32_t team_mask = distance_team_size == 32
        ? 0xffffffffu
        : (((1u << distance_team_size) - 1u) << team_start);

    for (uint32_t i = tid; i < queue_capacity; i += blockDim.x) {
        result_indices[i] = 0xFFFFFFFF;
        result_dists[i] = FLT_MAX;
    }
    __syncthreads();

    for (uint32_t i = team_id; i < num_pickup; i += num_teams) {
        uint32_t best_node_id = 0xFFFFFFFF;
        float best_dist = FLT_MAX;

        for (uint32_t sample = 0; sample < num_random_samplings; ++sample) {
            uint32_t gid = block_id + num_blocks * (i + num_pickup * sample);
            uint32_t node_id = 0xFFFFFFFF;
            if (seed_ptr != nullptr && gid < num_seeds_per_query) {
                node_id = seed_ptr[gid];
            }
            if (node_id >= num_dataset) {
                uint64_t x = (static_cast<uint64_t>(gid + 1) ^ rand_xor_mask);
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                node_id = (x * 0x2545F4914F6CDD1DULL) % num_dataset;
            }

            const float* node_ptr = dataset_ptr + static_cast<size_t>(node_id) * dim;
            float dist = compute_l2_for_team(query_buffer, node_ptr, dim, distance_team_size);
            if (dist < best_dist) {
                best_dist = dist;
                best_node_id = node_id;
            }
        }

        uint32_t node_id = best_node_id;

        bool usable = true;
        if (lane_in_team == 0) {
            usable = cagra::hashmap::insert(visited_hash, visited_hash_bitlen, node_id);
            if (usable && traversed_hash != nullptr) {
                usable = support_traversed_remove
                    ? !cagra::hashmap::search_support_remove(traversed_hash,
                                                             traversed_hash_bitlen,
                                                             node_id)
                    : !cagra::hashmap::search(traversed_hash, traversed_hash_bitlen, node_id);
            }
        }
        usable = __shfl_sync(team_mask, usable, team_start);

        if (lane_in_team == 0) {
            result_indices[i] = usable ? node_id : 0xFFFFFFFF;
            result_dists[i] = usable ? best_dist : FLT_MAX;
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
    uint64_t end_bucket,
    bool support_traversed_remove,
    uint32_t distance_team_size,
    bool skip_child_traversed_hash,
    unsigned long long* child_profile)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_in_warp = tid & 31u;
    const uint32_t lane_in_team = lane_in_warp & (distance_team_size - 1u);
    const uint32_t team_start = lane_in_warp & ~(distance_team_size - 1u);
    const uint32_t team_id = tid / distance_team_size;
    const uint32_t num_teams = blockDim.x / distance_team_size;
    const uint32_t team_mask = distance_team_size == 32
        ? 0xffffffffu
        : (((1u << distance_team_size) - 1u) << team_start);
    const uint32_t total_tasks = search_width * graph_degree;
    unsigned long long clk_graph = 0;
    unsigned long long clk_hash = 0;
    unsigned long long clk_dist = 0;
    unsigned long long clk_write = 0;

    for (uint32_t task_id = team_id; task_id < total_tasks; task_id += num_teams) {
        unsigned long long t_child = 0;
        uint32_t parent_idx;
        uint32_t neighbor_offset;
        if (graph_degree == 32) {
            parent_idx = task_id >> 5;
            neighbor_offset = task_id & 31u;
        } else if (graph_degree == 64) {
            parent_idx = task_id >> 6;
            neighbor_offset = task_id & 63u;
        } else {
            parent_idx = task_id / graph_degree;
            neighbor_offset = task_id % graph_degree;
        }
        uint32_t neighbor_id = 0xFFFFFFFF;
        bool usable = false;

        if (child_profile != nullptr && lane_in_team == 0) t_child = clock64();
        uint32_t parent_id = parent_list[parent_idx];
        if (parent_id != 0xFFFFFFFF) {
            neighbor_id = __ldg(knn_graph + static_cast<size_t>(parent_id) * graph_degree +
                                neighbor_offset);
            usable = neighbor_id != 0xFFFFFFFF;
        }

        if (usable && timestamps_ptr != nullptr) {
            uint64_t bucket_id = __ldg(&timestamps_ptr[neighbor_id]);
            usable = bucket_id >= start_bucket && bucket_id < end_bucket;
        }
        if (child_profile != nullptr && lane_in_team == 0) clk_graph += clock64() - t_child;

        if (child_profile != nullptr && lane_in_team == 0) t_child = clock64();
        if (lane_in_team == 0 && usable) {
            usable = cagra::hashmap::insert(visited_hash, visited_hash_bitlen, neighbor_id);
            if (usable && traversed_hash != nullptr && !skip_child_traversed_hash) {
                usable = support_traversed_remove
                    ? !cagra::hashmap::search_support_remove(traversed_hash,
                                                             traversed_hash_bitlen,
                                                             neighbor_id)
                    : !cagra::hashmap::search(traversed_hash, traversed_hash_bitlen, neighbor_id);
            }
        }
        if (child_profile != nullptr && lane_in_team == 0) clk_hash += clock64() - t_child;
        usable = __shfl_sync(team_mask, usable, team_start);

        float dist = FLT_MAX;
        if (child_profile != nullptr && lane_in_team == 0) t_child = clock64();
        if (usable) {
            const float* node_ptr = dataset_ptr + static_cast<size_t>(neighbor_id) * dim;
            dist = compute_l2_for_team(query_buffer, node_ptr, dim, distance_team_size);
        }
        if (child_profile != nullptr && lane_in_team == 0) clk_dist += clock64() - t_child;

        if (child_profile != nullptr && lane_in_team == 0) t_child = clock64();
        if (lane_in_team == 0) {
            candidate_indices[task_id] = usable ? neighbor_id : 0xFFFFFFFF;
            candidate_dists[task_id] = usable ? dist : FLT_MAX;
        }
        if (child_profile != nullptr && lane_in_team == 0) clk_write += clock64() - t_child;
    }
    if (child_profile != nullptr) {
        unsigned mask = __ballot_sync(0xffffffffu, lane_in_team == 0);
        for (int src = 0; src < 32; ++src) {
            if ((mask & (1u << src)) == 0) continue;
            unsigned long long graph = __shfl_sync(0xffffffffu, clk_graph, src);
            unsigned long long hash = __shfl_sync(0xffffffffu, clk_hash, src);
            unsigned long long dist = __shfl_sync(0xffffffffu, clk_dist, src);
            unsigned long long write = __shfl_sync(0xffffffffu, clk_write, src);
            if ((threadIdx.x & 31u) == 0) {
                atomicAdd(child_profile + 0, graph);
                atomicAdd(child_profile + 1, hash);
                atomicAdd(child_profile + 2, dist);
                atomicAdd(child_profile + 3, write);
            }
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
    uint32_t local_hash_bitlen,
    uint64_t start_bucket,
    uint64_t end_bucket,
    uint32_t* pre_hashmap,
    uint32_t* traversed_hashmap,
    uint32_t traversed_hash_bitlen,
    uint32_t queue_capacity,
    uint32_t distance_team_size,
    uint32_t num_random_samplings,
    bool use_bitonic_sorter,
    bool skip_child_traversed_hash,
    bool support_traversed_remove,
    unsigned long long* stage_profile)
{
    extern __shared__ uint8_t smem[];

    size_t offset = 0;
    float* query_buffer = reinterpret_cast<float*>(smem + offset);
    offset += (dim * sizeof(float) + 15) & ~15;

    uint32_t* visited_hash = nullptr;
    uint32_t linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    if (local_hash_bitlen < 14) {
        visited_hash = reinterpret_cast<uint32_t*>(smem + offset);
        offset += (((1u << local_hash_bitlen) * sizeof(uint32_t)) + 15) & ~15;
    } else {
        visited_hash = pre_hashmap + static_cast<size_t>(linear_block_id) * (1u << local_hash_bitlen);
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
    unsigned long long clk_sort = 0;
    unsigned long long clk_pickup = 0;
    unsigned long long clk_restore = 0;
    unsigned long long clk_child = 0;
    unsigned long long clk_cleanup = 0;

    if (query_id >= num_queries) return;

    const float* global_query = queries_ptr + static_cast<size_t>(query_id) * dim;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = global_query[i];
    }

    if (tid == 0) *terminate_flag = 0;
    cagra::hashmap::init(visited_hash, local_hash_bitlen);
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
        num_random_samplings,
        local_rand_mask,
        visited_hash,
        local_hash_bitlen,
        query_traversed_hash,
        traversed_hash_bitlen,
        block_id,
        num_blocks,
        support_traversed_remove,
        distance_team_size);
    __syncthreads();

    for (uint32_t iter = 0; iter < max_iterations; ++iter) {
        unsigned long long t_stage = 0;
        if (stage_profile != nullptr && tid == 0) t_stage = clock64();
        cagra::device::sort_multi_cta_buffer(
            result_dists, result_indices, queue_capacity, use_bitonic_sorter);
        if (tid == 0 && queue_capacity != 64 && queue_capacity != 128 &&
            queue_capacity != 256 && queue_capacity != 512) {
            printf("[multi_cta] unsupported queue_capacity=%u\n", queue_capacity);
        }
        __syncthreads();
        if (stage_profile != nullptr && tid == 0) clk_sort += clock64() - t_stage;

        if (stage_profile != nullptr && tid == 0) t_stage = clock64();
        if (tid < 32) {
            cagra::device::pickup_next_parent_multi_cta(
                const_cast<uint32_t*>(terminate_flag),
                parent_list,
                result_indices,
                result_dists,
                local_topk,
                search_width,
                query_traversed_hash,
                traversed_hash_bitlen,
                support_traversed_remove);
        }
        __syncthreads();
        if (stage_profile != nullptr && tid == 0) clk_pickup += clock64() - t_stage;

        if (*terminate_flag == 1) break;

        if (stage_profile != nullptr && tid == 0) t_stage = clock64();
        if (support_traversed_remove) {
            for (uint32_t i = tid; i < queue_capacity; i += blockDim.x) {
                uint32_t idx = result_indices[i];
                if (idx == 0xFFFFFFFF) continue;
                if (i >= local_topk && (idx & 0x80000000u) != 0u &&
                    query_traversed_hash != nullptr) {
                    cagra::hashmap::remove(query_traversed_hash,
                                           traversed_hash_bitlen,
                                           idx & 0x7FFFFFFFu);
                    result_indices[i] = 0xFFFFFFFF;
                    result_dists[i] = FLT_MAX;
                }
            }
            __syncthreads();
        }
        if (stage_profile != nullptr && tid == 0) clk_cleanup += clock64() - t_stage;

        if (stage_profile != nullptr && tid == 0) t_stage = clock64();
        cagra::hashmap::init(visited_hash, local_hash_bitlen);
        __syncthreads();
        for (uint32_t i = tid; i < queue_capacity; i += blockDim.x) {
            uint32_t idx = result_indices[i];
            if (idx == 0xFFFFFFFF) continue;
            cagra::hashmap::insert(visited_hash, local_hash_bitlen, idx & 0x7FFFFFFF);
        }
        __syncthreads();
        if (stage_profile != nullptr && tid == 0) clk_restore += clock64() - t_stage;

        if (stage_profile != nullptr && tid == 0) t_stage = clock64();
        unsigned long long* child_profile = stage_profile == nullptr
            ? nullptr
            : stage_profile + static_cast<size_t>(linear_block_id) * 10 + 6;
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
            local_hash_bitlen,
            query_traversed_hash,
            traversed_hash_bitlen,
            parent_list,
            search_width,
            start_bucket,
            end_bucket,
            support_traversed_remove,
            distance_team_size,
            skip_child_traversed_hash,
            child_profile);
        __syncthreads();
        if (stage_profile != nullptr && tid == 0) clk_child += clock64() - t_stage;

        if (stage_profile != nullptr && tid == 0) t_stage = clock64();
        if (support_traversed_remove) {
            for (uint32_t i = tid; i < local_topk; i += blockDim.x) {
                uint32_t idx = result_indices[i];
                if (idx == 0xFFFFFFFF || (idx & 0x80000000u) != 0u) continue;
                if (query_traversed_hash != nullptr &&
                    cagra::hashmap::search_support_remove(query_traversed_hash,
                                                          traversed_hash_bitlen,
                                                          idx)) {
                    result_indices[i] = 0xFFFFFFFF;
                    result_dists[i] = FLT_MAX;
                }
            }
            __syncthreads();
        }
        if (stage_profile != nullptr && tid == 0) clk_cleanup += clock64() - t_stage;
    }

    unsigned long long clk_final_sort = 0;
    if (stage_profile != nullptr && tid == 0) clk_final_sort = clock64();
    cagra::device::sort_multi_cta_buffer(
        result_dists, result_indices, queue_capacity, use_bitonic_sorter);
    __syncthreads();
    if (stage_profile != nullptr && tid == 0) clk_final_sort = clock64() - clk_final_sort;

    size_t out_base =
        (static_cast<size_t>(query_id) * num_cta + cta_id) * static_cast<size_t>(local_topk);
    for (uint32_t i = tid; i < local_topk; i += blockDim.x) {
        uint32_t idx = result_indices[i];
        bool valid = idx != 0xFFFFFFFFu;
        idx &= 0x7FFFFFFFu;
        if (valid && query_traversed_hash != nullptr && (result_indices[i] & 0x80000000u) == 0u) {
            valid = support_traversed_remove
                ? cagra::hashmap::insert_support_remove(query_traversed_hash,
                                                        traversed_hash_bitlen,
                                                        idx)
                : cagra::hashmap::insert(query_traversed_hash, traversed_hash_bitlen, idx);
        }
        intermediate_indices[out_base + i] = valid ? idx : 0xFFFFFFFFu;
        intermediate_dists[out_base + i] = valid ? result_dists[i] : FLT_MAX;
    }

    if (stage_profile != nullptr && tid == 0) {
        const size_t base = static_cast<size_t>(linear_block_id) * 10;
        atomicAdd(stage_profile + base + 0, clk_sort);
        atomicAdd(stage_profile + base + 1, clk_pickup);
        atomicAdd(stage_profile + base + 2, clk_restore);
        atomicAdd(stage_profile + base + 3, clk_child);
        atomicAdd(stage_profile + base + 4, clk_cleanup);
        atomicAdd(stage_profile + base + 5, clk_final_sort);
    }
}

__global__ void merge_multi_cta_results_kernel(
    uint32_t* intermediate_indices,
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

    extern __shared__ uint8_t smem[];
    float* best_dists = reinterpret_cast<float*>(smem);
    uint32_t* best_positions = reinterpret_cast<uint32_t*>(best_dists + blockDim.x);

    const size_t base = static_cast<size_t>(query_id) * num_cta * local_topk;
    const size_t out_base = static_cast<size_t>(query_id) * topk;
    const uint32_t num_candidates = num_cta * local_topk;

    for (uint32_t out_k = 0; out_k < topk; ++out_k) {
        float local_best_dist = FLT_MAX;
        uint32_t local_best_pos = 0xFFFFFFFF;

        for (uint32_t i = threadIdx.x; i < num_candidates; i += blockDim.x) {
            uint32_t idx = intermediate_indices[base + i];
            float dist = intermediate_dists[base + i];
            if (idx == 0xFFFFFFFF) continue;

            bool duplicate = false;
            for (uint32_t prev = 0; prev < out_k; ++prev) {
                if (out_indices[out_base + prev] == static_cast<int64_t>(idx)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate &&
                (dist < local_best_dist ||
                 (dist == local_best_dist && i < local_best_pos))) {
                local_best_dist = dist;
                local_best_pos = i;
            }
        }

        best_dists[threadIdx.x] = local_best_dist;
        best_positions[threadIdx.x] = local_best_pos;
        __syncthreads();

        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                float other_dist = best_dists[threadIdx.x + stride];
                uint32_t other_pos = best_positions[threadIdx.x + stride];
                if (other_dist < best_dists[threadIdx.x] ||
                    (other_dist == best_dists[threadIdx.x] &&
                     other_pos < best_positions[threadIdx.x])) {
                    best_dists[threadIdx.x] = other_dist;
                    best_positions[threadIdx.x] = other_pos;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            size_t dst = out_base + out_k;
            uint32_t best_pos = best_positions[0];
            if (best_pos == 0xFFFFFFFF) {
                out_indices[dst] = -1;
                out_dists[dst] = FLT_MAX;
            } else {
                out_indices[dst] = static_cast<int64_t>(intermediate_indices[base + best_pos]);
                out_dists[dst] = best_dists[0];
                intermediate_indices[base + best_pos] = 0xFFFFFFFF;
            }
        }
        __syncthreads();
    }
}

} // namespace device

uint32_t resolve_multi_cta_local_topk()
{
    uint32_t local_topk = 32;
    if (const char* local_topk_env = std::getenv("CAGRA_MULTI_CTA_LOCAL_TOPK")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(local_topk_env, nullptr, 10));
        if (requested == 32 || requested == 64 || requested == 128 || requested == 256) {
            local_topk = requested;
        }
    }
    return local_topk;
}

uint32_t resolve_multi_cta_count(int64_t k, SearchParams params, uint32_t num_cta_per_query)
{
    uint32_t local_topk = resolve_multi_cta_local_topk();
    uint32_t topk = static_cast<uint32_t>(k);
    uint32_t global_itopk = std::max(topk, params.itopk_size);
    if (num_cta_per_query == 0) {
        num_cta_per_query = std::max(params.search_width, (global_itopk + local_topk - 1) / local_topk);
        num_cta_per_query = std::max(num_cta_per_query, 12u);
    }
    if (const char* min_cta_env = std::getenv("CAGRA_MULTI_CTA_MIN_CTA")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(min_cta_env, nullptr, 10));
        if (requested > 0) {
            num_cta_per_query = std::max(num_cta_per_query, requested);
        }
    }
    if (const char* max_cta_env = std::getenv("CAGRA_MULTI_CTA_MAX_CTA")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(max_cta_env, nullptr, 10));
        if (requested > 0) {
            num_cta_per_query = std::min(num_cta_per_query, requested);
        }
    }
    return std::max(1u, num_cta_per_query);
}

size_t multi_cta_intermediate_count(int64_t num_queries, uint32_t num_cta_per_query)
{
    uint32_t local_topk = resolve_multi_cta_local_topk();
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
                                       bool run_merge,
                                       uint64_t* d_stage_profile)
{
    if (d_graph == nullptr) {
        throw std::runtime_error("Graph is null!");
    }

    uint32_t local_topk = resolve_multi_cta_local_topk();
    uint32_t local_search_width = 1;
    if (const char* local_width_env = std::getenv("CAGRA_MULTI_CTA_LOCAL_WIDTH")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(local_width_env, nullptr, 10));
        if (requested >= 1 && requested <= 8) {
            local_search_width = requested;
        }
    }
    bool support_traversed_remove = false;
    if (const char* cleanup_env = std::getenv("CAGRA_MULTI_CTA_CUVS_CLEANUP")) {
        support_traversed_remove = std::strtoul(cleanup_env, nullptr, 10) != 0;
    }
    bool use_bitonic_sorter = true;
    if (const char* sorter_env = std::getenv("CAGRA_MULTI_CTA_SORTER")) {
        std::string sorter = sorter_env;
        use_bitonic_sorter = sorter != "cub";
    }
    bool skip_child_traversed_hash = false;
    if (const char* skip_env = std::getenv("CAGRA_MULTI_CTA_SKIP_CHILD_TRAVERSED")) {
        skip_child_traversed_hash = std::strtoul(skip_env, nullptr, 10) != 0;
    }
    num_cta_per_query = resolve_multi_cta_count(k, params, num_cta_per_query);

    uint32_t raw_needed = local_topk + local_search_width * graph_degree;
    uint32_t queue_capacity = cagra::detail::next_power_of_2(raw_needed);
    queue_capacity = std::min(queue_capacity, 512u);
    if (queue_capacity < 64) queue_capacity = 64;

    uint32_t local_hash_bitlen = std::min(params.hash_bitlen, 12u);
    if (const char* local_hash_env = std::getenv("CAGRA_MULTI_CTA_LOCAL_HASH_BITLEN")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(local_hash_env, nullptr, 10));
        if (requested >= 8 && requested <= params.hash_bitlen) {
            local_hash_bitlen = requested;
        }
    }

    uint32_t distance_team_size = 32;
    if (dim <= 128) {
        distance_team_size = 8;
    } else if (dim <= 256) {
        distance_team_size = 16;
    }
    if (const char* team_env = std::getenv("CAGRA_MULTI_CTA_TEAM_SIZE")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(team_env, nullptr, 10));
        if (requested == 4 || requested == 8 || requested == 16 || requested == 32) {
            distance_team_size = requested;
        }
    }
    uint32_t num_random_samplings = 1;
    if (const char* random_env = std::getenv("CAGRA_MULTI_CTA_RANDOM_SAMPLINGS")) {
        uint32_t requested = static_cast<uint32_t>(std::strtoul(random_env, nullptr, 10));
        if (requested >= 1 && requested <= 16) {
            num_random_samplings = requested;
        }
    }

    auto align16 = [](size_t bytes) { return (bytes + 15) & ~size_t{15}; };
    size_t smem_size = 0;
    smem_size += align16(static_cast<size_t>(dim) * sizeof(float));
    if (local_hash_bitlen < 14) {
        smem_size += align16((size_t{1} << local_hash_bitlen) * sizeof(uint32_t));
    }
    smem_size += align16(static_cast<size_t>(queue_capacity) * sizeof(uint32_t));
    smem_size += align16(static_cast<size_t>(queue_capacity) * sizeof(float));
    smem_size += align16(static_cast<size_t>(local_search_width) * sizeof(uint32_t));
    smem_size += 16;

    size_t hash_count = multi_cta_hash_count(num_queries, num_cta_per_query, local_hash_bitlen);
    size_t traversed_hash_count = multi_cta_traversed_hash_count(num_queries, params.hash_bitlen);
    if (d_stage_profile != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(d_stage_profile,
                                   0,
                                   static_cast<size_t>(num_queries) * num_cta_per_query * 10 *
                                       sizeof(uint64_t),
                                   stream));
    }
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
        local_hash_bitlen,
        start_bucket,
        end_bucket,
        d_pre_hashmap,
        d_traversed_hashmap,
        params.hash_bitlen,
        queue_capacity,
        distance_team_size,
        num_random_samplings,
        use_bitonic_sorter,
        skip_child_traversed_hash,
        support_traversed_remove,
        reinterpret_cast<unsigned long long*>(d_stage_profile));
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
                                local_topk,
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
                             uint32_t local_topk,
                             cudaStream_t stream,
                             float* profile_ms)
{
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    if (profile_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
    }

    constexpr uint32_t merge_block_size = 256;
    size_t merge_smem_size = merge_block_size * (sizeof(float) + sizeof(uint32_t));
    cagra::device::merge_multi_cta_results_kernel<<<static_cast<uint32_t>(num_queries),
                                                     merge_block_size,
                                                     merge_smem_size,
                                                     stream>>>(
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
