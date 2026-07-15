#!/usr/bin/env python3
"""
Pareto frontier evaluation for SeRF and HNSW.

The script evaluates multiple index and search configurations on DEEP-96,
SIFT-128, GIST-960, and WIT-2048. Index construction, ground-truth
generation, and querying use multiple threads. Existing indexes and result
rows are reused when available.

The output CSV contains method, dataset, M, K, K_Search, range_pct, recall,
qps, comps, build_time, and index_size_mb.
"""

import subprocess
import os
import sys
import pandas as pd
from datetime import datetime
import time

# ============== CONFIGURATION ==============

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SERF_ROOT = os.path.join(PROJECT_ROOT, "third_party", "SeRF")

# Executable paths
SERF_BINARY = os.path.join(SERF_ROOT, "build/benchmark/serf_multithread")
HNSW_BINARY = os.path.join(SERF_ROOT, "build/benchmark/hnsw_multithread")

# Index and result directories
INDEX_DIR = os.path.join(PROJECT_ROOT, "results", "exp1_search_efficiency", "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp1_search_efficiency")

# Leap strategy used by SeRF
STRATEGY = "MAX_POS"

# Dataset configuration
DATASETS = {
    "DEEP-96": {
        "base_path": os.environ.get("GRAB_DEEP_BASE", ""),
        "query_path": os.environ.get("GRAB_DEEP_QUERY", ""),
        "dataset_name": "deep",
    },
    "SIFT-128": {
        "base_path": os.environ.get("GRAB_SIFT_BASE", ""),
        "query_path": os.environ.get("GRAB_SIFT_QUERY", ""),
        "dataset_name": "local",
    },
    "GIST-960": {
        "base_path": os.environ.get("GRAB_GIST_BASE", ""),
        "query_path": os.environ.get("GRAB_GIST_QUERY", ""),
        "dataset_name": "local",
    },
    "WIT-2048": {
        "base_path": os.environ.get("GRAB_WIT_BASE", ""),
        "query_path": os.environ.get("GRAB_WIT_QUERY", ""),
        "dataset_name": "local",
    },
}

# All datasets use one million vectors.
DATA_SIZE = 1000000

# Evaluated methods
METHODS = ["hnsw", "serf"]

# Index configurations used to construct the Pareto frontier
INDEX_CONFIGS = [
    {"M": 8,  "ef_con": 100},
    {"M": 8,  "ef_con": 200},
    {"M": 16, "ef_con": 200},
    {"M": 16, "ef_con": 400},
    {"M": 32, "ef_con": 400},
    {"M": 32, "ef_con": 800},
    {"M": 64, "ef_con": 400},
    {"M": 64, "ef_con": 800},
]

# Search configurations used to evaluate the recall-QPS trade-off
EF_SEARCH_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

# Evaluated range percentages
RANGE_PCTS = [1.0, 10.0, 20.0, 100.0]

# OpenMP thread configuration
OMP_THREADS_BUILD = 30      # Threads used for index construction
OMP_THREADS_QUERY = 30      # Threads used for ground-truth generation

# Query worker threads
NUM_QUERY_THREADS = 30

# ============== UTILITIES ==============

def get_index_path(method, dataset_key, m, k):
    """Return the index path for one configuration."""
    dataset_short = dataset_key.split('-')[0].lower()  # DEEP-96 -> deep
    return os.path.join(INDEX_DIR, f"{method}_{dataset_short}_1m_M{m}_K{k}.bin")

def index_exists(method, dataset_key, m, k):
    """Return whether a nonempty index file exists."""
    path = get_index_path(method, dataset_key, m, k)
    return os.path.exists(path) and os.path.getsize(path) > 0

def get_index_size_mb(method, dataset_key, m, k):
    """Return the index file size in MiB."""
    path = get_index_path(method, dataset_key, m, k)
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return None

def get_binary(method):
    """Return the executable path for a method."""
    return SERF_BINARY if method == "serf" else HNSW_BINARY

def get_existing_results(csv_path):
    """Read completed configurations from an existing result file."""
    if not os.path.exists(csv_path):
        return set()

    try:
        df = pd.read_csv(csv_path)
        # Each key is (method, dataset, M, K, K_Search).
        existing = set()
        for _, row in df.iterrows():
            key = (row['method'], row['dataset'], row['M'], row['K'], row['K_Search'])
            existing.add(key)
        return existing
    except Exception as e:
        print(f"[WARN] Could not read existing results from {csv_path}: {e}")
        return set()

# ============== TASK GENERATION ==============

def generate_tasks(csv_path):
    """Generate all index-construction and query tasks."""
    tasks = []
    existing_results = get_existing_results(csv_path)

    for dataset_key, dataset_info in DATASETS.items():
        for method in METHODS:
            for config in INDEX_CONFIGS:
                m = config["M"]
                k = config["ef_con"]

                # Check whether this index configuration is already available.
                has_index = index_exists(method, dataset_key, m, k)
                index_size = get_index_size_mb(method, dataset_key, m, k) if has_index else None

                # Check whether every K_Search value has been evaluated.
                all_tested = True
                for k_search in EF_SEARCH_VALUES:
                    key = (method, dataset_key, m, k, k_search)
                    if key not in existing_results:
                        all_tested = False
                        break

                tasks.append({
                    "method": method,
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_info["dataset_name"],
                    "base_path": dataset_info["base_path"],
                    "query_path": dataset_info["query_path"],
                    "M": m,
                    "K": k,
                    "index_path": get_index_path(method, dataset_key, m, k),
                    "has_index": has_index,
                    "index_size_mb": index_size,
                    "all_tested": all_tested,
                })

    return tasks

# ============== TASK EXECUTION ==============

def run_build(task):
    """Run one index-construction task."""
    method = task["method"]
    dataset_key = task["dataset_key"]
    m = task["M"]
    k = task["K"]

    index_path = task["index_path"]
    binary = get_binary(method)

    # Configure OpenMP for index construction.
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS_BUILD)

    # Index-construction command
    cmd = [
        binary,
        "-dataset", task["dataset_name"],
        "-N", str(DATA_SIZE),
        "-dataset_path", task["base_path"],
        "-query_path", task["query_path"],
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search_list", ",".join(map(str, EF_SEARCH_VALUES)),
        "-save_index", index_path,
        "-threads", str(NUM_QUERY_THREADS),  # Query worker threads
    ]

    # SeRF-specific argument
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    print(f"\n[BUILD] {method} {dataset_key} M={m} K={k}")

    output_file = f"/tmp/{method}_{dataset_key}_M{m}_K{k}_build.txt"

    try:
        os.makedirs(INDEX_DIR, exist_ok=True)

        t0 = time.time()
        with open(output_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      env=env, universal_newlines=True, bufsize=1)

            # Stream subprocess output to both the terminal and a temporary file.
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        build_time = time.time() - t0

        # Record the index size after construction.
        index_size_mb = get_index_size_mb(method, dataset_key, m, k)

        print(f"  [DONE] build_time={build_time:.2f}s, size={index_size_mb:.2f}MB")

        return {
            "build_time": build_time,
            "index_size_mb": index_size_mb,
        }

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)

def run_query(task, existing_results, build_time=None, index_size_mb=None):
    """Evaluate K_Search values that are not present in the result file."""
    method = task["method"]
    dataset_key = task["dataset_key"]
    m = task["M"]
    k = task["K"]

    # Reused indexes do not have construction timing in this run.
    if build_time is None:
        build_time = 0.0
    if index_size_mb is None:
        index_size_mb = task.get("index_size_mb", 0.0)

    index_path = task["index_path"]
    binary = get_binary(method)

    # Querying requires the saved index.
    if not os.path.exists(index_path):
        print(f"[SKIP] {method} {dataset_key}: index file not found")
        return []

    # Select K_Search values that have not been evaluated.
    untested_k_search = []
    for k_search in EF_SEARCH_VALUES:
        key = (method, dataset_key, m, k, k_search)
        if key not in existing_results:
            untested_k_search.append(k_search)

    if not untested_k_search:
        print(f"[SKIP] {method} {dataset_key} M={m} K={k}: all K_Search values completed")
        return []

    # Configure OpenMP for ground-truth generation.
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS_QUERY)

    # Query command for the remaining K_Search values
    cmd = [
        binary,
        "-dataset", task["dataset_name"],
        "-N", str(DATA_SIZE),
        "-dataset_path", task["base_path"],
        "-query_path", task["query_path"],
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search_list", ",".join(map(str, untested_k_search)),
        "-load_index", index_path,
        "-threads", str(NUM_QUERY_THREADS),  # Query worker threads
    ]

    # SeRF-specific argument
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    print(f"\n[QUERY] {method} {dataset_key} M={m} K={k} ef_search={untested_k_search}")

    output_file = f"/tmp/{method}_{dataset_key}_M{m}_K{k}_query.txt"

    try:
        t0 = time.time()
        with open(output_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      env=env, universal_newlines=True, bufsize=1)

            # Stream subprocess output to both the terminal and a temporary file.
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        query_time = time.time() - t0

        # Parse benchmark output.
        results = []
        current_k_search = None

        with open(output_file, 'r') as f:
            for line in f:
                # Track the K_Search value associated with following result rows.
                if "Testing with search_ef=" in line or "search_ef=" in line:
                    try:
                        current_k_search = int(line.split("=")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                    continue

                if line.startswith("range:"):
                    parts = line.split()
                    if len(parts) >= 8:
                        range_val = int(parts[1])
                        recall = float(parts[3])
                        qps = float(parts[5])
                        comps = float(parts[7])
                        range_pct = round((range_val * 100) / DATA_SIZE, 1)

                        # Retain the range percentages defined by the protocol.
                        if range_pct in RANGE_PCTS:
                            results.append({
                                'method': method,
                                'dataset': dataset_key,
                                'M': m,
                                'K': k,
                                'K_Search': current_k_search,
                                'range_pct': range_pct,
                                'recall': recall,
                                'qps': qps,
                                'comps': comps,
                                'build_time': build_time,
                                'index_size_mb': index_size_mb,
                            })

        print(f"  [DONE] {len(results)} results in {query_time:.2f}s")
        return results

    except Exception as e:
        print(f"  [ERROR] {e}")
        return []
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)

def run_single_task(task, csv_path):
    """Construct an index when needed, then run its query configurations."""
    method = task["method"]
    dataset_key = task["dataset_key"]
    m = task["M"]
    k = task["K"]

    print(f"\n{'='*80}")
    print(f"[TASK] {method} {dataset_key} M={m} K={k}")
    print(f"{'='*80}")

    # Refresh completed configurations before starting the task.
    existing_results = get_existing_results(csv_path)

    # Initialize construction metadata for a reused index.
    build_time = 0.0
    index_size_mb = task.get("index_size_mb", 0.0)

    # Construct the index if it is not already available.
    if not task["has_index"]:
        build_result = run_build(task)
        if build_result is None:
            print("[FAIL] Index construction failed; query task skipped")
            return []
        build_time = build_result["build_time"]
        index_size_mb = build_result["index_size_mb"]
    else:
        print(f"[INFO] Reusing existing index: {index_size_mb:.2f}MB")

    # Evaluate configurations that are not already in the result file.
    query_results = run_query(task, existing_results, build_time, index_size_mb)

    return query_results

# ============== MAIN ==============

def main():
    # Create output directories.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # SeRF and HNSW write to the same result file.
    UNIFIED_CSV = os.path.join(RESULTS_DIR, "pareto_all.csv")
    CSV_PATH = UNIFIED_CSV

    print("="*80)
    print("Pareto Frontier Test for SeRF/HNSW")
    print("="*80)
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Dataset size: {DATA_SIZE//1000000}M")
    print(f"Methods: {METHODS}")
    print(f"Index configurations: {len(INDEX_CONFIGS)}")
    print(f"Search parameters: ef_search = {EF_SEARCH_VALUES}")
    print(f"Range: {RANGE_PCTS}%")
    print("="*80)
    print(f"Index-construction threads: {OMP_THREADS_BUILD}")
    print(f"Query threads: {NUM_QUERY_THREADS}")
    print(f"Ground-truth threads: {OMP_THREADS_QUERY}")
    print("="*80)

    # Generate tasks.
    tasks = generate_tasks(CSV_PATH)

    print(f"\nTotal tasks: {len(tasks)}")

    # Summarize task status.
    tasks_with_index = sum(1 for t in tasks if t["has_index"])
    tasks_all_tested = sum(1 for t in tasks if t["all_tested"])
    tasks_need_build = sum(1 for t in tasks if not t["has_index"])
    tasks_need_query = sum(1 for t in tasks if not t["all_tested"])

    print(f"  - Existing indexes: {tasks_with_index}/{len(tasks)}")
    print(f"  - Indexes to build: {tasks_need_build}/{len(tasks)}")
    print(f"  - Completed tasks: {tasks_all_tested}/{len(tasks)}")
    print(f"  - Query tasks remaining: {tasks_need_query}/{len(tasks)}")

    # Create the result file if needed.
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write("method,dataset,M,K,K_Search,range_pct,recall,qps,comps,build_time,index_size_mb\n")
        print(f"[INFO] Created result file: {CSV_PATH}")
    else:
        print(f"[INFO] Using existing result file: {CSV_PATH}")

    # Execute tasks sequentially.
    completed = 0
    for task_idx, task in enumerate(tasks, 1):
        print(f"\n[PROGRESS] Task {task_idx}/{len(tasks)}")

        results = run_single_task(task, CSV_PATH)

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

    # Display result counts by dataset and method.
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print("\nResult rows by dataset and method:")
        summary = df.groupby(['dataset', 'method']).size().reset_index(name='count')
        print(summary.to_string(index=False))

    return 0

if __name__ == "__main__":
    sys.exit(main())
