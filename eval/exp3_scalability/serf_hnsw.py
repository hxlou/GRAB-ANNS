#!/usr/bin/env python3
"""
DEEP10M scalability evaluation for SeRF and HNSW.

The evaluation uses fixed parameters M=32, K=400, and K_Search=256 while
scaling the dataset from 1M to 10M vectors in 1M increments. It records index
construction time, index size, recall, and QPS for ranges covering 1%, 10%,
20%, and 100% of the dataset. Tasks are executed sequentially.
"""

import subprocess
import os
import sys
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== CONFIGURATION ==============

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SERF_ROOT = os.path.join(PROJECT_ROOT, "third_party", "SeRF")

# Executable paths
SERF_BINARY = os.path.join(SERF_ROOT, "build/benchmark/serf_multithread")
HNSW_BINARY = os.path.join(SERF_ROOT, "build/benchmark/hnsw_multithread")

# Index and result directories
INDEX_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability", "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability")

# DEEP base and query files
DEEP_BASE_PATH = os.environ.get("GRAB_DEEP10M_BASE", os.environ.get("GRAB_DEEP_BASE", ""))
DEEP_QUERY_PATH = os.environ.get("GRAB_DEEP_QUERY", "")

# Leap strategy used by SeRF
STRATEGY = "MAX_POS"

# Fixed index and search parameters
FIXED_M = 32
FIXED_K = 400
FIXED_K_SEARCH = 256

# Dataset sizes from 1M to 10M in 1M increments
DATA_SIZES = [1000000 * i for i in range(1, 11)]

# Evaluated range percentages
RANGE_PCTS = [1.0, 10.0, 20.0, 100.0]

# Evaluated methods
METHODS = ["hnsw", "serf"]

# OpenMP thread configuration
OMP_THREADS_BUILD = 30      # Threads used for index construction
OMP_THREADS_QUERY = 30      # Threads used for ground-truth generation

# Query worker threads
NUM_QUERY_THREADS = 30

# Number of concurrently scheduled evaluation tasks
WORKERS = 1

# ============== UTILITIES ==============

def get_index_path(method, data_size, m, k):
    """Return the index path for one dataset size and configuration."""
    size_suffix = f"{data_size//1000000}m"
    return os.path.join(INDEX_DIR, f"{method}_deep_{size_suffix}_M{m}_K{k}.bin")

def index_exists(method, data_size, m, k):
    """Return whether a nonempty index file exists."""
    path = get_index_path(method, data_size, m, k)
    return os.path.exists(path) and os.path.getsize(path) > 0

def get_index_size_mb(method, data_size, m, k):
    """Return the index file size in MiB."""
    path = get_index_path(method, data_size, m, k)
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return None

def get_binary(method):
    """Return the executable path for a method."""
    return SERF_BINARY if method == "serf" else HNSW_BINARY

# ============== TASK GENERATION ==============

def generate_tasks():
    """Generate all index-construction and query tasks."""
    tasks = []

    for size in DATA_SIZES:
        for method in METHODS:
            # Check whether the index is already available.
            has_index = index_exists(method, size, FIXED_M, FIXED_K)
            index_size = get_index_size_mb(method, size, FIXED_M, FIXED_K) if has_index else None

            tasks.append({
                "method": method,
                "data_size": size,
                "M": FIXED_M,
                "K": FIXED_K,
                "K_Search": FIXED_K_SEARCH,
                "dataset_path": DEEP_BASE_PATH,
                "query_path": DEEP_QUERY_PATH,
                "index_path": get_index_path(method, size, FIXED_M, FIXED_K),
                "has_index": has_index,
                "index_size_mb": index_size,
            })

    return tasks

# ============== TASK EXECUTION ==============

def run_build(task):
    """Run one index-construction task."""
    method = task["method"]
    size = task["data_size"]
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
        "-dataset", "deep",
        "-N", str(size),
        "-dataset_path", task["dataset_path"],
        "-query_path", task["query_path"],
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search_list", str(FIXED_K_SEARCH),
        "-save_index", index_path,
    ]

    # SeRF-specific argument
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    size_str = f"{size//1000000}M"

    print(f"[BUILD] {method} deep_{size_str} M={m} K={k} ef_search={FIXED_K_SEARCH}")

    output_file = f"/tmp/{method}_deep_{size_str}_M{m}_K{k}_build.txt"

    try:
        os.makedirs(INDEX_DIR, exist_ok=True)

        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # Parse construction time from benchmark output.
        build_time = None
        with open(output_file, 'r') as f:
            for line in f:
                if "Build" in line and "Time" in line and "Index" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        time_str = parts[-1].rstrip('s')
                        try:
                            build_time = float(time_str)
                        except ValueError:
                            pass
                    break

        # Record the index file size.
        index_size_mb = get_index_size_mb(method, size, m, k)

        if build_time is not None:
            print(f"  [DONE] build_time={build_time:.2f}s, size={index_size_mb:.2f}MB")
        else:
            print(f"  [DONE] size={index_size_mb:.2f}MB")

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

def run_query(task):
    """Run one query task."""
    method = task["method"]
    size = task["data_size"]
    m = task["M"]
    k = task["K"]

    index_path = task["index_path"]
    binary = get_binary(method)

    # Querying requires the saved index.
    if not os.path.exists(index_path):
        print(f"[SKIP] {method} deep_{size//1000000}M: index file not found")
        return []

    # Configure OpenMP for ground-truth generation.
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS_QUERY)

    # Query command
    cmd = [
        binary,
        "-dataset", "deep",
        "-N", str(size),
        "-dataset_path", task["dataset_path"],
        "-query_path", task["query_path"],
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search_list", str(FIXED_K_SEARCH),
        "-load_index", index_path,
        "-threads", str(NUM_QUERY_THREADS),  # Query worker threads
    ]

    # SeRF-specific argument
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    size_str = f"{size//1000000}M"

    print(f"[QUERY] {method} deep_{size_str} M={m} K={k} ef_search={FIXED_K_SEARCH}")

    output_file = f"/tmp/{method}_deep_{size_str}_M{m}_K{k}_query.txt"

    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # Parse benchmark output.
        results = []
        current_k_search = None

        with open(output_file, 'r') as f:
            for line in f:
                # Track the K_Search value associated with following rows.
                if "Testing with search_ef=" in line:
                    current_k_search = int(line.split("=")[1].strip().split()[0])
                    continue

                if line.startswith("range:"):
                    parts = line.split()
                    if len(parts) >= 8:
                        range_val = int(parts[1])
                        recall = float(parts[3])
                        qps = float(parts[5])
                        comps = float(parts[7])
                        range_pct = round((range_val * 100) / size, 1)

                        # Retain the range percentages defined by the protocol.
                        if range_pct in RANGE_PCTS:
                            results.append({
                                'method': method,
                                'dataset': f"DEEP-{size//1000000}M",
                                'M': m,
                                'K': k,
                                'K_Search': current_k_search,
                                'range_pct': range_pct,
                                'recall': recall,
                                'qps': qps,
                                'comps': comps,
                            })

        print(f"  [DONE] {len(results)} results (range_pcts={RANGE_PCTS})")
        return results

    except Exception as e:
        print(f"  [ERROR] {e}")
        return []
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)

def run_single_task(task):
    """Construct an index when needed, then run its query task."""
    method = task["method"]
    size = task["data_size"]
    size_str = f"{size//1000000}M"

    print(f"\n{'='*80}")
    print(f"[TASK] {method} deep_{size_str} M={FIXED_M} K={FIXED_K} K_Search={FIXED_K_SEARCH}")
    print(f"{'='*80}")

    results = []
    build_time = None
    index_size_mb = task.get("index_size_mb")

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

    # Run the query task.
    query_results = run_query(task)

    # Attach construction metadata to query results.
    for r in query_results:
        r["build_time"] = build_time if build_time is not None else 0
        r["size_mb"] = index_size_mb

    return query_results

# ============== MAIN ==============

def main():
    # Create output directories.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*80)
    print("DEEP10M Fixed-Parameter Scalability Evaluation")
    print("="*80)
    print(f"Dataset: {DEEP_BASE_PATH}")
    print(f"Query:  {DEEP_QUERY_PATH}")
    print(f"Dataset sizes: {[f'{s//1000000}M' for s in DATA_SIZES]}")
    print(f"Fixed parameters: M={FIXED_M}, K={FIXED_K}, K_Search={FIXED_K_SEARCH}")
    print(f"Range: {RANGE_PCTS}%")
    print(f"Methods: {METHODS}")
    print("="*80)
    print(f"Index-construction threads: {OMP_THREADS_BUILD}")
    print(f"Query threads: {NUM_QUERY_THREADS}")
    print(f"Ground-truth threads: {OMP_THREADS_QUERY}")
    print(f"Concurrent tasks: {WORKERS}")
    print("="*80)

    # Generate tasks.
    tasks = generate_tasks()

    print(f"\nTotal tasks: {len(tasks)}")
    print("Task list:")
    for task in tasks:
        status = "existing index" if task["has_index"] else "index construction required"
        print(f"  - {task['method']} deep_{task['data_size']//1000000}M: {status}")

    # Execute tasks.
    all_results = []
    result_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {executor.submit(run_single_task, task): task for task in tasks}

        for future in as_completed(future_to_task):
            results = future.result()
            with result_lock:
                all_results.extend(results)
            completed += 1
            print(f"\n[PROGRESS] {completed}/{len(tasks)}")

    # Save results.
    if all_results:
        df = pd.DataFrame(all_results)
        result_path = os.path.join(RESULTS_DIR, f"fixed_params_m32_k400_ef256_{TIMESTAMP}.csv")
        df.to_csv(result_path, index=False)

        print("\n" + "="*80)
        print(f"Results saved to: {result_path}")
        print(f"Result rows: {len(all_results)}")
        print("="*80)

        # Display result counts by dataset, method, and range percentage.
        print("\nResult rows by dataset, method, and range percentage:")
        summary = df.groupby(['dataset', 'method', 'range_pct']).size().reset_index(name='count')
        print(summary.to_string(index=False))

        # Display the 10% range summary.
        print("\nSummary (range=10%):")
        print("-" * 80)
        print(f"{'Method':<10} {'Dataset':<10} {'BuildTime(s)':<12} {'Size(MB)':<12} {'Recall':<10} {'QPS':<12}")
        print("-" * 80)

        for dataset in sorted(df['dataset'].unique(), key=lambda x: int(x.replace('DEEP-', '').replace('M', ''))):
            for method in METHODS:
                subset = df[(df['dataset'] == dataset) &
                           (df['method'] == method) &
                           (df['range_pct'] == 10.0)]
                if len(subset) > 0:
                    row = subset.iloc[0]
                    bt = f"{row['build_time']:.2f}" if row['build_time'] > 0 else "N/A"
                    sz = f"{row['size_mb']:.2f}" if row['size_mb'] else "N/A"
                    print(f"{method:<10} {dataset:<10} {bt:<12} {sz:<12} {row['recall']:<10.4f} {row['qps']:<12.1f}")

        print("-" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
