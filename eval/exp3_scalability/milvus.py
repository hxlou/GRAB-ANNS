#!/usr/bin/env python3
"""
DEEP10M scalability evaluation for Milvus HNSW.

The evaluation sweeps the configured index and search parameters over the
selected dataset sizes and range percentages. It records index construction
time, estimated index size, recall, and QPS. Results are appended after each
task, and the process uses the configured CPU affinity.
"""

import time
import sys
import os
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# Flush progress output line by line.
sys.stdout.reconfigure(line_buffering=True)

# Force fork start method to avoid copying large numpy arrays per worker.
try:
    mp.set_start_method("fork", force=True)
except (RuntimeError, ValueError):
    pass

# ============== CONFIGURATION ==============

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# CPU affinity - Pin to efficiency cores (16-31)
# Ground-truth generation temporarily uses all configured CPU cores.
E_CORES = list(range(16, 32))

def set_cpu_affinity():
    """Set process CPU affinity to efficiency cores"""
    try:
        os.sched_setaffinity(0, E_CORES)
        print(f"[CPU Affinity] Process pinned to efficiency cores: {E_CORES}")
    except AttributeError:
        print("[WARNING] os.sched_setaffinity not available")
    except Exception as e:
        print(f"[WARNING] Failed to set CPU affinity: {e}")

def release_cpu_affinity():
    """Allow the process to run on all available CPU cores."""
    try:
        all_cores = list(range(cpu_count()))
        os.sched_setaffinity(0, all_cores)
        print(f"[CPU Affinity] Released to all cores: {all_cores}")
    except AttributeError:
        print("[WARNING] os.sched_setaffinity not available")
    except Exception as e:
        print(f"[WARNING] Failed to release CPU affinity: {e}")

def get_current_affinity():
    """Return the current CPU affinity."""
    try:
        return os.sched_getaffinity(0)
    except AttributeError:
        return E_CORES
    except Exception as e:
        return E_CORES

# Milvus connection
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = "19530"

# Milvus reconnection wait (seconds)
MILVUS_WAIT_TIMEOUT_SEC = int(os.environ.get("MILVUS_WAIT_TIMEOUT_SEC", "300"))
MILVUS_WAIT_INTERVAL_SEC = int(os.environ.get("MILVUS_WAIT_INTERVAL_SEC", "5"))

# Milvus operation retries (for transient etcd/IO stalls)
MILVUS_OP_RETRIES = 3
MILVUS_OP_RETRY_SLEEP = 10

# Task retries after a Milvus crash/restart
TASK_RETRIES = 2

def ensure_milvus_connection(max_retries=5, sleep_sec=2):
    """Ensure Milvus connection is alive."""
    for attempt in range(max_retries):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=30, overwrite=True)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[CONNECT] Retry {attempt+1}/{max_retries} after error: {e}")
                time.sleep(sleep_sec)
            else:
                print(f"[ERROR] Failed to connect to Milvus: {e}")
                return False

def wait_for_milvus(timeout_sec=MILVUS_WAIT_TIMEOUT_SEC, interval_sec=MILVUS_WAIT_INTERVAL_SEC):
    """Wait for Milvus to come back after a crash/restart."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if ensure_milvus_connection(max_retries=1, sleep_sec=0):
            return True
        print(f"[WAIT] Milvus unavailable, retry in {interval_sec}s...")
        time.sleep(interval_sec)
    return False

def milvus_call(op_name, func, *args, **kwargs):
    """Retry Milvus operations on transient errors."""
    for attempt in range(MILVUS_OP_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "connection refused" in err_str.lower() or "failed to connect to all addresses" in err_str.lower():
                if wait_for_milvus():
                    continue
            if attempt < MILVUS_OP_RETRIES - 1:
                print(f"[WARN] {op_name} failed, retry {attempt+1}/{MILVUS_OP_RETRIES}: {e}")
                time.sleep(MILVUS_OP_RETRY_SLEEP)
                ensure_milvus_connection()
            else:
                raise

set_cpu_affinity()

# Output directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ground-truth cache directory
GT_CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability", "gt_cache")
os.makedirs(GT_CACHE_DIR, exist_ok=True)

# DEEP base and query files
DEEP_BASE_PATH = os.environ.get("GRAB_DEEP10M_BASE", os.environ.get("GRAB_DEEP_BASE", ""))
DEEP_QUERY_PATH = os.environ.get("GRAB_DEEP_QUERY", "")

# Index configurations
INDEX_CONFIGS = [
    # {"M": 8, "K": 100},
    # {"M": 8, "K": 200},
    # {"M": 16, "K": 100},
    # {"M": 16, "K": 200},
    # {"M": 32, "K": 200},
    {"M": 32, "K": 400},
    # {"M": 64, "K": 100},
    # {"M": 64, "K": 200},
    # {"M": 64, "K": 400},
]

# Search configurations; ef must be at least TOP_K.
K_SEARCH_VALUES = [16, 32, 64, 128, 256, 512, 1024]

# Evaluated dataset sizes
DATA_SIZES = [1000000 * i for i in range(10, 11)]

# Evaluated range percentages
RANGE_PCTS = [10.0]

# CSV column order
CSV_COLUMNS = [
    "method", "dataset", "M", "K", "K_Search", "range_pct",
    "recall", "qps", "comps", "build_time", "size_mb",
]

# Query configuration
NUM_QUERIES = 1000  # Number of queries in the batch evaluation
TOP_K = 10
METRIC_TYPE = "L2"
DIM = 96  # DEEP vector dimension

# Batch insert size
BATCH_SIZE = 1000

# Number of queries per Milvus request
BATCH_QUERY_SIZE = 200  # 200 queries per batch

# ============== GT CACHE FUNCTIONS ==============

# Read-only arrays shared by forked workers
_shared_vectors = None
_shared_queries = None

def get_gt_cache_path(size):
    """Return the ground-truth cache path for a dataset size."""
    size_str = f"{size//1000000}m"
    return os.path.join(GT_CACHE_DIR, f"deep_{size_str}_gt_cache.pkl")

def _compute_single_query_gt(args):
    """Compute ground truth for one query using shared arrays."""
    query_idx, l_bound, r_bound, top_k = args
    # Access read-only arrays inherited by forked workers.
    range_vectors = _shared_vectors[l_bound:r_bound+1]
    query_vector = _shared_queries[query_idx:query_idx+1]
    distances = np.sum((query_vector - range_vectors) ** 2, axis=1)
    k_actual = min(top_k, len(range_vectors))
    if k_actual > 0:
        kth_idx = k_actual - 1
        gt_indices_rel = np.argpartition(distances, kth_idx)[:k_actual]
        gt_ids = set(l_bound + gt_indices_rel)
    else:
        gt_ids = set()
    return (query_idx, gt_ids)

def precompute_ground_truth(size, range_pcts, num_queries=NUM_QUERIES, num_ranges_per_pct=5, verbose=True):
    """Precompute and cache ground truth with multiple worker processes.

    The cache maps each range percentage to tuples containing the lower bound,
    upper bound, and ground-truth IDs for every query. Each query is computed
    independently for every generated range.
    """
    cache_path = get_gt_cache_path(size)

    # Reuse an existing cache for this dataset size.
    if os.path.exists(cache_path):
        if verbose:
            print(f"[GT CACHE] Found existing cache: {cache_path}")
            print(f"[GT CACHE] Loading precomputed GT...")
        with open(cache_path, 'rb') as f:
            gt_cache = pickle.load(f)
        if verbose:
            print(f"[GT CACHE] Loaded {len(gt_cache)} range_pct values from cache")
        return gt_cache

    # Generate the cache with forked workers and read-only shared arrays.
    num_workers = 30

    if verbose:
        print(f"\n[GT PRECOMPUTE] Generating GT for size={size//1000000}M (Multi-threaded)")
        print(f"[GT PRECOMPUTE] Using {num_workers} workers with shared data (no copy)...")

    global _shared_vectors, _shared_queries
    _shared_vectors = load_fvecs(DEEP_BASE_PATH, size)
    _shared_queries = load_fvecs(DEEP_QUERY_PATH, num_queries)
    actual_size = _shared_vectors.shape[0]

    gt_cache = {}

    with Pool(processes=num_workers) as pool:
        for range_pct in range_pcts:
            if verbose:
                print(f"  [GT PRECOMPUTE] Range={range_pct}%...", end=" ", flush=True)

            range_list = []

            for _ in range(num_ranges_per_pct):
                # Generate a range.
                range_width = max(1, int(actual_size * range_pct / 100))
                max_l_bound = actual_size - range_width
                l_bound = np.random.randint(0, max(1, max_l_bound))
                r_bound = l_bound + range_width - 1

                # Pass query indices and scalar bounds to workers.
                query_tasks = [(query_idx, l_bound, r_bound, TOP_K) for query_idx in range(num_queries)]

                # Compute query ground truth in worker processes.
                results = pool.map(_compute_single_query_gt, query_tasks)

                # Restore query order after parallel processing.
                results.sort(key=lambda x: x[0])
                all_gt_ids = [gt_ids for _, gt_ids in results]

                range_list.append((l_bound, r_bound, all_gt_ids))

            gt_cache[range_pct] = range_list

            if verbose:
                print(f"Done ({len(range_list)} ranges, {num_queries} queries each)")

    _shared_vectors = None
    _shared_queries = None

    # Persist the completed cache.
    if verbose:
        print(f"[GT CACHE] Saving to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(gt_cache, f)

    if verbose:
        print(f"[GT CACHE] Saved successfully")

    return gt_cache

# ============== UTILITY FUNCTIONS ==============

def load_fvecs(path, count=None):
    """Load fvecs file format"""
    with open(path, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if count is None:
            data = np.frombuffer(f.read(), dtype=np.float32)
        else:
            bytes_to_read = int(count) * int(dim) * 4
            data = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
        return data.reshape(-1, dim)

def get_collection_name(size, m, ef_con):
    """Return the collection name for an index configuration."""
    size_str = f"{size//1000000}m"
    return f"deep_{size_str}_M{m}_EFCON{ef_con}"

def collection_has_index(collection):
    """Return whether a collection has an index."""
    indexes = collection.indexes
    return len(indexes) > 0

# ============== INDEX SIZE ESTIMATION ==============

def estimate_index_size(size, m, ef_con):
    """
    Estimate HNSW index size from graph links, levels, vectors, and IDs.

    The estimate uses 2M links per node, eight bytes per link, four bytes for
    the level, DIM * 4 bytes for the vector, and eight bytes for the ID.
    """
    # Approximate number of links per node
    edges_per_node = m * 2

    # Bytes per link for neighbor ID and link data
    bytes_per_edge = 4 + 4

    # Estimated bytes per node
    bytes_per_node = (
        edges_per_node * bytes_per_edge +  # edges
        4 +                                  # level
        DIM * 4 +                            # vector
        8                                    # id
    )

    total_bytes = size * bytes_per_node
    return total_bytes / (1024 * 1024)  # Convert to MiB.

# ============== BUILD FUNCTION ==============

def build_collection(size, m, ef_con, verbose=True):
    """Create a collection, insert vectors, and construct its HNSW index.

    build_time measures create_index only; insertion time is reported
    separately and is not included in that field.
    """
    if not ensure_milvus_connection():
        raise RuntimeError("Milvus connection unavailable during build.")
    collection_name = get_collection_name(size, m, ef_con)

    if verbose:
        print(f"\n{'='*80}")
        print(f"[BUILD] Milvus deep_{size//1000000}M M={m} ef_con={ef_con}")
        print(f"{'='*80}")
        print(f"  Collection: {collection_name}")

    # Drop an existing collection so each task constructs a new index.
    if utility.has_collection(collection_name):
        if verbose:
            print(f"  [CLEANUP] Dropping existing collection to ensure fresh build...")
        utility.drop_collection(collection_name)
        if verbose:
            print(f"  [CLEANUP] Dropped existing collection")

    # Create the collection.
    if verbose:
        print(f"  [CREATE] Creating new collection...")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields, description=f"DEEP-{size//1000000}M M={m} ef_con={ef_con}")
    collection = Collection(name=collection_name, schema=schema)

    # Insert vectors before the timed create_index call.
    if verbose:
        print(f"  [INSERT] Loading and inserting {size} vectors...")
        print("    Insertion time is reported separately from build_time")

    vectors = load_fvecs(DEEP_BASE_PATH, size)
    actual_size = vectors.shape[0]
    ids = np.arange(actual_size)

    insert_start = time.time()
    for i in range(0, actual_size, BATCH_SIZE):
        end = min(i + BATCH_SIZE, actual_size)
        milvus_call("insert", collection.insert, [ids[i:end].tolist(), vectors[i:end]])

        # Report insertion progress.
        if verbose and (i // BATCH_SIZE) % 10 == 0:
            progress = (end / actual_size) * 100
            print(f"    Insert progress: {progress:.1f}% ({end}/{actual_size})")

    collection.flush()
    insert_time = time.time() - insert_start

    if verbose:
        print(f"  [INSERT] Inserted {actual_size} vectors in {insert_time:.2f}s")

    # Time the create_index operation.
    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": "HNSW",
        "params": {"M": m, "efConstruction": ef_con}
    }

    if verbose:
        print("  [INDEX] Building HNSW index")
        print(f"    Dataset size: {size:,} vectors")
        print(f"    Parameters: M={m}, efConstruction={ef_con}")

    build_start = time.time()
    milvus_call("create_index", collection.create_index, field_name="embedding", index_params=index_params)
    build_time = time.time() - build_start

    # Estimate index size.
    index_size_mb = estimate_index_size(size, m, ef_con)

    if verbose:
        print(f"  [INDEX] Index built in {build_time:.2f}s ({build_time/60:.1f} min)")
        print(f"  [SIZE]  Estimated index size: {index_size_mb:.2f} MB")
        print(f"  [TIMING] Insert: {insert_time:.2f}s, build_time: {build_time:.2f}s")

    # Load the collection for querying.
    if verbose:
        print(f"  [LOAD] Loading collection into memory...")

    milvus_call("load", collection.load)

    if verbose:
        print(f"  [DONE] Collection ready for query")

    return {
        "collection_name": collection_name,
        "build_time": build_time,
        "index_size_mb": index_size_mb,
    }

# ============== QUERY FUNCTION ==============

def query_collection(collection_name, ef, range_pcts, gt_cache, queries, verbose=True):
    """Run batched queries using cached ground truth.

    Args:
        collection_name: Milvus collection name.
        ef: Search parameter.
        range_pcts: Range percentages to evaluate.
        gt_cache: Precomputed ground-truth cache.
        queries: Loaded query vectors.
    """
    if not ensure_milvus_connection():
        raise RuntimeError("Milvus connection unavailable during query.")
    collection = Collection(collection_name)

    if verbose:
        print(f"\n[QUERY] Testing ef={ef}, ranges={range_pcts}")
        print(f"  [BATCH] Queries per request: {BATCH_QUERY_SIZE}")
        print("  [GT CACHE] Using precomputed ground truth")

    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": ef}}

    results = []
    total_tests = len(range_pcts)
    current_test = 0

    for range_pct in range_pcts:
        current_test += 1

        if verbose:
            print(f"  [{current_test}/{total_tests}] Range={range_pct}%", end=" ", flush=True)

        # Read ranges and exact results from the cache.
        range_list = gt_cache[range_pct]

        total_recall = 0.0
        total_qps = 0.0
        valid_range_count = 0

        for range_idx, (l_bound, r_bound, all_gt_ids) in enumerate(range_list):
            expr = f"id >= {l_bound} && id <= {r_bound}"

            # Warm up the batched query path.
            if range_idx == 0:
                try:
                    collection.search(
                        data=queries[:10].astype(np.float32),
                        anns_field="embedding",
                        param=search_params,
                        limit=TOP_K,
                        expr=expr,
                        output_fields=["id"]
                    )
                except:
                    pass

            # Submit queries in batches of BATCH_QUERY_SIZE.
            num_queries = queries.shape[0]
            t_start = time.time()
            batch_recall = 0.0
            valid_count = 0

            for start in range(0, num_queries, BATCH_QUERY_SIZE):
                end = min(start + BATCH_QUERY_SIZE, num_queries)
                batch_queries = queries[start:end].astype(np.float32)
                res = milvus_call("search", collection.search,
                    data=batch_queries,
                    anns_field="embedding",
                    param=search_params,
                    limit=TOP_K,
                    expr=expr,
                    output_fields=["id"]
                )

                for i in range(end - start):
                    query_idx = start + i
                    result_ids = set(int(hit.id) for hit in res[i])
                    gt_ids = all_gt_ids[query_idx]

                    k_denom = min(TOP_K, len(gt_ids)) if gt_ids else TOP_K
                    query_recall = len(result_ids & gt_ids) / k_denom if k_denom > 0 else 0.0
                    batch_recall += query_recall
                    valid_count += 1

            t_end = time.time()

            # Compute QPS for this range.
            batch_qps = num_queries / (t_end - t_start) if (t_end - t_start) > 0 else 0
            total_qps += batch_qps

            avg_recall = batch_recall / valid_count if valid_count > 0 else 0.0
            total_recall += avg_recall
            valid_range_count += 1

        avg_recall = total_recall / valid_range_count if valid_range_count > 0 else 0.0
        avg_qps = total_qps / valid_range_count if valid_range_count > 0 else 0.0

        results.append({
            'range_pct': range_pct,
            'recall': avg_recall,
            'qps': avg_qps,
        })

        if verbose:
            print(f"-> recall={avg_recall:.4f}, qps={avg_qps:.1f}")

    return results

# ============== SINGLE TASK FUNCTION ==============

def run_single_task(size, m, ef_con, ef_list, gt_cache, queries):
    """Construct one index and evaluate its search configurations.

    Args:
        size: Dataset size.
        m, ef_con: Index parameters.
        ef_list: Search parameter values.
        gt_cache: Precomputed ground-truth cache.
        queries: Loaded query vectors.
    """
    size_str = f"{size//1000000}M"
    collection_name = get_collection_name(size, m, ef_con)

    print(f"\n{'='*80}")
    print(f"[TASK] Milvus deep_{size_str} M={m} ef_con={ef_con} ef_list={ef_list}")
    print(f"{'='*80}")

    # Index-construction phase
    build_result = build_collection(size, m, ef_con, verbose=True)

    results = []
    for ef in ef_list:
        # Query phase using cached ground truth
        query_results = query_collection(
            collection_name=build_result["collection_name"],
            ef=ef,
            range_pcts=RANGE_PCTS,
            gt_cache=gt_cache,
            queries=queries,
            verbose=True
        )

        # Assemble result rows.
        for qr in query_results:
            results.append({
                'method': 'milvus',
                'dataset': f'DEEP-{size_str}',
                'M': m,
                'K': ef_con,
                'K_Search': ef,
                'range_pct': qr['range_pct'],
                'recall': qr['recall'],
                'qps': qr['qps'],
                'comps': 0,
                'build_time': build_result['build_time'],
                'size_mb': build_result['index_size_mb'],
            })

    # Release the in-memory collection while retaining the index.
    try:
        collection = Collection(build_result["collection_name"])
        collection.release()
        print(f"\n[RELEASE] Collection released from memory (index kept on disk)")
    except:
        pass

    return results

# ============== MAIN ==============

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"milvus_scaling_{timestamp}.csv")

    print("="*80)
    print("DEEP10M Milvus Scalability Evaluation")
    print("="*80)
    print(f"Dataset: {DEEP_BASE_PATH}")
    print(f"Query:  {DEEP_QUERY_PATH}")
    print(f"Dataset sizes: {[f'{s//1000000}M' for s in DATA_SIZES]}")
    print(f"Index configurations: {INDEX_CONFIGS}")
    print(f"Search parameters (ef): {K_SEARCH_VALUES}")
    print(f"Range: {RANGE_PCTS}%")
    print("="*80)

    # Connect to Milvus.
    print("\n[CONNECT] Connecting to Milvus...")
    if not ensure_milvus_connection():
        return 1
    print(f"[CONNECT] Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    # List existing collections.
    all_collections = utility.list_collections()
    print(f"\n[INFO] Existing collections: {len(all_collections)}")
    for coll in all_collections:
        if coll.startswith("deep_"):
            print(f"  - {coll}")

    # Initialize the append-only result file.
    csv_header = ",".join(CSV_COLUMNS)
    with open(result_path, 'w') as f:
        f.write(csv_header + '\n')

    # Load query vectors once for all tasks.
    print("\n[QUERIES] Loading query vectors...")
    queries = load_fvecs(DEEP_QUERY_PATH, NUM_QUERIES)
    print(f"[QUERIES] Loaded {queries.shape[0]} queries")

    # Execute tasks grouped by dataset size.
    total_tasks = len(DATA_SIZES) * len(INDEX_CONFIGS)
    all_results = []
    first_save = False

    task_num = 0
    for size in DATA_SIZES:
        size_str = f"{size//1000000}M"

        # Precompute ground truth for this dataset size.
        print(f"\n\n{'='*80}")
        print(f"[GT PRECOMPUTE] Dataset size: {size_str}")
        print(f"{'='*80}")

        # Temporarily use all CPU cores for ground-truth generation.
        original_affinity = get_current_affinity()
        print(f"[CPU Affinity] Saved affinity: {original_affinity}")
        print("[CPU Affinity] Using all cores for ground-truth generation")
        release_cpu_affinity()

        try:
            gt_cache = precompute_ground_truth(size, RANGE_PCTS, num_queries=NUM_QUERIES, verbose=True)
        finally:
            # Restore the configured affinity after ground-truth generation.
            print(f"[CPU Affinity] Restoring configured CPU set: {E_CORES}")
            set_cpu_affinity()

        # Reuse the ground-truth cache for all index configurations.
        for cfg in INDEX_CONFIGS:
            task_num += 1
            m = cfg["M"]
            ef_con = cfg["K"]

            print(f"\n\n{'#'*80}")
            print(f"# Task {task_num}/{total_tasks} - {size_str} M={m} ef_con={ef_con}")
            print(f"{'#'*80}")

            try:
                attempt = 0
                while True:
                    try:
                        results = run_single_task(size, m, ef_con, K_SEARCH_VALUES, gt_cache, queries)
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt >= TASK_RETRIES:
                            raise
                        print(f"\n[WARN] Task failed (attempt {attempt}/{TASK_RETRIES}): {e}")
                        if not wait_for_milvus():
                            raise RuntimeError("Milvus did not recover in time.")

                if results:
                    all_results.extend(results)

                    # Append completed result rows.
                    df_new = pd.DataFrame(results, columns=CSV_COLUMNS)
                    df_new.to_csv(result_path, mode='a', header=first_save, index=False)

                    print(f"\n[PROGRESS] {task_num}/{total_tasks} | saved rows: {len(all_results)}")

            except Exception as e:
                print(f"\n[ERROR] Task failed: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    if all_results:
        df = pd.DataFrame(all_results)

        print("\n" + "="*80)
        print("Evaluation completed")
        print("="*80)
        print(f"Result file: {result_path}")
        print(f"Result rows: {len(all_results)}")

        # Display the 10% range summary.
        print("\nSummary (range=10%):")
        print("-" * 80)
        print(f"{'Method':<10} {'Dataset':<10} {'BuildTime(s)':<12} {'Size(MB)':<12} "
              f"{'Recall':<10} {'QPS':<12}")
        print("-" * 80)

        for size in DATA_SIZES:
            dataset = f"DEEP-{size//1000000}M"
            subset = df[(df['dataset'] == dataset) & (df['range_pct'] == 10.0)]
            if len(subset) > 0:
                row = subset.iloc[0]
                bt = f"{row['build_time']:.2f}" if row['build_time'] > 0 else "N/A"
                sz = f"{row['size_mb']:.2f}" if row['size_mb'] else "N/A"
                print(f"{'milvus':<10} {dataset:<10} {bt:<12} {sz:<12} "
                      f"{row['recall']:<10.4f} {row['qps']:<12.1f}")

        print("-" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
