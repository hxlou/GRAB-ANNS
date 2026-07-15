#!/usr/bin/env python3
"""
Pareto frontier evaluation for Milvus HNSW.

The script evaluates the same datasets, query files, index parameters, search
parameters, and output schema used by the SeRF and HNSW evaluation. Existing
collections and result rows are reused when available. The process is pinned
to the configured CPU set.
"""

import time
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# ============== CONFIGURATION ==============

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Milvus connection
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# Output directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp1_search_efficiency")

# CPU affinity used for this process
E_CORES = list(range(16, 32))

def set_cpu_affinity():
    """Pin the process to the configured CPU set."""
    try:
        os.sched_setaffinity(0, E_CORES)
        print(f"[CPU] Affinity set to: {E_CORES}")
    except AttributeError:
        print("[WARN] os.sched_setaffinity is unavailable")
    except Exception as e:
        print(f"[WARN] Could not set CPU affinity: {e}")

# Apply CPU affinity during initialization.
set_cpu_affinity()

# Dataset configuration shared with SeRF and HNSW
DATASETS = {
    "DEEP-96": {
        "base_path": os.environ.get("GRAB_DEEP_BASE", ""),
        "query_path": os.environ.get("GRAB_DEEP_QUERY", ""),
        "dim": 96,
    },
    "SIFT-128": {
        "base_path": os.environ.get("GRAB_SIFT_BASE", ""),
        "query_path": os.environ.get("GRAB_SIFT_QUERY", ""),
        "dim": 128,
    },
    "GIST-960": {
        "base_path": os.environ.get("GRAB_GIST_BASE", ""),
        "query_path": os.environ.get("GRAB_GIST_QUERY", ""),
        "dim": 960,
    },
    "WIT-2048": {
        "base_path": os.environ.get("GRAB_WIT_BASE", ""),
        "query_path": os.environ.get("GRAB_WIT_QUERY", ""),
        "dim": 2048,
    },
}

# Dataset size
DATA_SIZE = 1000000

# Index parameters
INDEX_CONFIGS = [
    {"M": 8,  "ef_con": 100},
    {"M": 8,  "ef_con": 200},
    {"M": 16, "ef_con": 200},
    {"M": 16, "ef_con": 400},
    {"M": 32, "ef_con": 200},
    {"M": 32, "ef_con": 400},
    {"M": 64, "ef_con": 100},
    # {"M": 64, "ef_con": 200},
]

# Search parameters
EF_SEARCH_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Evaluated range percentages
RANGE_PCTS = [1.0, 10.0, 20.0, 100.0]

# Query configuration
NUM_QUERIES = 1000
TOP_K = 10
QUERY_SEED = 42

# Milvus index configuration
INDEX_TYPE = "HNSW"
METRIC_TYPE = "L2"

# ============== UTILITIES ==============

def load_fvecs(path, count=None):
    """Load vectors from an fvecs file."""
    with open(path, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if count is None:
            data = np.frombuffer(f.read(), dtype=np.float32)
        else:
            bytes_to_read = int(count) * int(dim) * 4
            data = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
        return data.reshape(-1, dim)

def get_batch_size(dim):
    """Select an insertion batch size within the gRPC message limit."""
    if dim >= 2048:
        return 7500
    elif dim >= 960:
        return 16000
    elif dim >= 128:
        return 120000
    else:
        return 150000

def synthesize_queries(vectors, num_queries=NUM_QUERIES, seed=QUERY_SEED):
    """Generate synthetic queries with the same procedure as the C++ runner."""
    n, dim = vectors.shape
    queries = np.zeros((num_queries, dim), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for i in range(num_queries):
        random_indices = rng.integers(0, n, size=dim)
        for j in range(dim):
            queries[i, j] = vectors[random_indices[j], j]

    return queries

def get_collection_name(dataset_key, m, ef_con):
    """Return the collection name for an index configuration."""
    dataset_short = dataset_key.split('-')[0].lower()
    return f"{dataset_short}_1m_M{m}_EFCON{ef_con}"

def get_existing_results(csv_path):
    """Read completed configurations from an existing result file."""
    if not os.path.exists(csv_path):
        return set()

    try:
        df = pd.read_csv(csv_path)
        existing = set()
        for _, row in df.iterrows():
            key = (row['method'], row['dataset'], row['M'], row['K'], row['K_Search'])
            existing.add(key)
        return existing
    except Exception as e:
        print(f"[WARN] Could not read existing results: {e}")
        return set()

# ============== MILVUS OPERATIONS ==============

def create_collection_if_needed(collection_name, dim, vectors, m, ef_con):
    """Create and populate a collection when it is not already available."""
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        num_entities = collection.num_entities

        if num_entities == DATA_SIZE:
            print(f"  [INFO] Reusing collection: {collection_name} ({num_entities} vectors)")
            return collection
        else:
            print(f"  [WARN] Collection is incomplete: {collection_name} ({num_entities}/{DATA_SIZE})")
            print("  [INFO] Dropping and recreating the collection")
            utility.drop_collection(collection_name)

    # Create a collection.
    print(f"  [CREATE] Collection: {collection_name}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description=f"M={m}, ef_con={ef_con}")
    collection = Collection(name=collection_name, schema=schema)

    # Insert vectors in dimension-dependent batches.
    actual_size = vectors.shape[0]
    ids = np.arange(actual_size)
    batch_size = get_batch_size(dim)

    print(f"  [INSERT] Vectors: {actual_size}")
    for i in range(0, actual_size, batch_size):
        end = min(i + batch_size, actual_size)
        collection.insert([ids[i:end].tolist(), vectors[i:end]])
    collection.flush()
    print(f"  [INSERT] Completed: {actual_size} vectors")

    return collection

def build_index_if_needed(collection, m, ef_con):
    """Construct the index when it is not already available."""
    indexes = collection.indexes
    if indexes:
        print("  [INFO] Reusing existing index")
        # Verify that the existing index can be loaded.
        try:
            collection.load()
            collection.release()
            return True, 0.0
        except Exception as e:
            print(f"  [WARN] Index is not ready: {e}")
            return False, 0.0

    # Construct the HNSW index.
    print(f"  [INDEX] HNSW parameters: M={m}, efConstruction={ef_con}")

    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"M": m, "efConstruction": ef_con}
    }

    t0 = time.time()
    collection.create_index(field_name="embedding", index_params=index_params)
    build_time = time.time() - t0

    # Verify that the constructed index can be loaded.
    try:
        collection.load()
        collection.release()
        print(f"  [INDEX] Completed: {build_time:.2f}s")
        return True, build_time
    except Exception as e:
        print(f"  [ERROR] Index construction failed: {e}")
        return False, build_time

def test_range_query_batch(collection, queries, vectors, ef, range_pct):
    """Run range-filtered queries and compute recall."""
    actual_size = vectors.shape[0]
    range_width = int(actual_size * range_pct / 100)
    range_width = max(1, range_width)

    # Generate query ranges.
    rng = np.random.default_rng(QUERY_SEED)
    if range_pct == 100:
        l_bounds = np.zeros(NUM_QUERIES, dtype=int)
    else:
        max_l_bound = actual_size - range_width - 80
        max_l_bound = max(0, max_l_bound)
        l_bounds = rng.integers(0, max_l_bound + 1, size=NUM_QUERIES)

    # Compute exact ground truth.
    total_recall = 0.0
    total_time = 0.0

    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": ef}}

    collection.load()  # Ensure that the collection is loaded.

    for query_idx in range(NUM_QUERIES):
        l_bound = int(l_bounds[query_idx])
        r_bound = l_bound + range_width - 1

        # Ground truth
        range_vectors = vectors[l_bound:r_bound+1]
        query_vector = queries[query_idx:query_idx+1]

        # Compute distances without allocating a full difference matrix.
        vec_sq = np.einsum('ij,ij->i', range_vectors, range_vectors)
        query_sq = np.einsum('i,i', query_vector[0], query_vector[0])
        dot_prod = np.dot(range_vectors, query_vector.flatten())
        distances = query_sq + vec_sq - 2 * dot_prod

        k_actual = min(TOP_K, len(range_vectors))
        if k_actual > 0:
            gt_indices_rel = np.argpartition(distances, k_actual-1)[:k_actual]
            gt_ids = set(l_bound + gt_indices_rel)
        else:
            gt_ids = set()

        # Run the Milvus query.
        expr = f"id >= {l_bound} && id <= {r_bound}"

        t0 = time.time()
        res = collection.search(
            data=query_vector,
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            expr=expr,
            output_fields=["id"]
        )
        query_time = time.time() - t0

        # Compute recall.
        result_ids = set()
        for hit in res[0]:
            if hasattr(hit, 'id'):
                result_ids.add(int(hit.id))

        recall = len(result_ids & gt_ids) / len(gt_ids) if gt_ids else 0.0
        total_recall += recall
        total_time += query_time

    avg_recall = total_recall / NUM_QUERIES
    avg_qps = NUM_QUERIES / total_time if total_time > 0 else 0.0

    return avg_recall, avg_qps

# ============== MAIN WORKFLOW ==============

def run_single_task(task, csv_path, existing_results):
    """Run one index and search configuration."""
    dataset_key = task["dataset_key"]
    dataset_info = task["dataset_info"]
    m = task["M"]
    ef_con = task["K"]

    collection_name = get_collection_name(dataset_key, m, ef_con)

    print(f"\n{'='*80}")
    print(f"[TASK] {dataset_key} M={m} K={ef_con}")
    print(f"{'='*80}")

    # Select ef_search values that have not been evaluated.
    untested_ef = []
    for ef in EF_SEARCH_VALUES:
        key = ('milvus', dataset_key, m, ef_con, ef)
        if key not in existing_results:
            untested_ef.append(ef)

    if not untested_ef:
        print("[SKIP] All ef_search values completed")
        return []

    print(f"[INFO] Remaining ef_search values: {untested_ef}")

    # Load base vectors.
    print(f"[LOAD] Base vectors: {dataset_info['base_path']}")
    vectors = load_fvecs(dataset_info['base_path'], DATA_SIZE)
    print(f"[LOAD] Vector array shape: {vectors.shape}")

    # Load the query file or generate queries when no file is available.
    query_path = dataset_info['query_path']
    if query_path and os.path.exists(query_path):
        queries = load_fvecs(query_path, NUM_QUERIES)
        print(f"[LOAD] Query array shape: {queries.shape}")
    else:
        print(f"[SYNTH] Generating {NUM_QUERIES} queries")
        queries = synthesize_queries(vectors, NUM_QUERIES)

    # Create or reuse the collection.
    collection = create_collection_if_needed(collection_name, dataset_info['dim'], vectors, m, ef_con)

    # Construct or reuse the index.
    index_ready, build_time = build_index_if_needed(collection, m, ef_con)

    if not index_ready:
        print("[ERROR] Index is not ready; query task skipped")
        return []

    # Estimate collection size from float32 vectors and index overhead.
    data_size_mb = (DATA_SIZE * dataset_info['dim'] * 4) / (1024 * 1024)
    index_size_mb = data_size_mb * 1.5  # Estimated index overhead

    # Run query configurations.
    results = []

    for ef in untested_ef:
        for range_pct in RANGE_PCTS:
            print(f"\n[TEST] ef={ef}, range={range_pct}%", end=" ", flush=True)

            recall, qps = test_range_query_batch(collection, queries, vectors, ef, range_pct)

            print(f"-> recall={recall:.3f}, qps={qps:.1f}")

            results.append({
                'method': 'milvus',
                'dataset': dataset_key,
                'M': m,
                'K': ef_con,
                'K_Search': ef,
                'range_pct': range_pct,
                'recall': recall,
                'qps': qps,
                'comps': 0,
                'build_time': build_time,
                'index_size_mb': index_size_mb,
            })

    # Release the collection after the task.
    collection.release()
    print("[INFO] Collection released")

    return results

def generate_tasks():
    """Generate all evaluation tasks."""
    tasks = []

    for dataset_key, dataset_info in DATASETS.items():
        for config in INDEX_CONFIGS:
            m = config["M"]
            ef_con = config["ef_con"]

            tasks.append({
                "dataset_key": dataset_key,
                "dataset_info": dataset_info,
                "M": m,
                "K": ef_con,
            })

    return tasks

def main():
    # Create the output directory.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Milvus, SeRF, and HNSW write to the same result file.
    UNIFIED_CSV = os.path.join(RESULTS_DIR, "pareto_all.csv")
    CSV_PATH = UNIFIED_CSV

    print("="*80)
    print("Pareto Frontier Test for Milvus")
    print("="*80)
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Dataset size: {DATA_SIZE//1000000}M")
    print(f"Index configurations: {len(INDEX_CONFIGS)}")
    print(f"Search parameters: ef_search = {EF_SEARCH_VALUES}")
    print(f"Range: {RANGE_PCTS}%")
    print("="*80)

    # Connect to Milvus.
    print("\n[CONNECT] Milvus")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=60)
        print(f"[CONNECT] Connected: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("[HINT] Start the Milvus service before running this evaluation")
        return 1

    # Generate tasks.
    tasks = generate_tasks()
    print(f"\nTotal tasks: {len(tasks)}")

    # Create the result file if needed.
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write("method,dataset,M,K,K_Search,range_pct,recall,qps,comps,build_time,index_size_mb\n")
        print(f"[INFO] Created result file: {CSV_PATH}")
    else:
        print(f"[INFO] Using existing result file: {CSV_PATH}")

    # Read completed configurations.
    existing_results = get_existing_results(CSV_PATH)
    print(f"Completed configurations: {len(existing_results)}")

    # Execute tasks sequentially.
    completed = 0
    for task_idx, task in enumerate(tasks, 1):
        print(f"\n[PROGRESS] Task {task_idx}/{len(tasks)}")

        results = run_single_task(task, CSV_PATH, existing_results)

        # Append each completed task to the result file.
        if results:
            with open(CSV_PATH, 'a') as f:
                for r in results:
                    row = f"{r['method']},{r['dataset']},{r['M']},{r['K']},{r['K_Search']},{r['range_pct']},{r['recall']},{r['qps']},{r['comps']},{r['build_time']},{r['index_size_mb']}\n"
                    f.write(row)
            completed += len(results)

    print("\n" + "="*80)
    print("Evaluation completed")
    print(f"Results saved to: {CSV_PATH}")
    print(f"New result rows: {completed}")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
