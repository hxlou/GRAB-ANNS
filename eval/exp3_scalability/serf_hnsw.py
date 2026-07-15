#!/usr/bin/env python3
"""
DEEP10M 固定参数规模扩展测试脚本

固定参数: M=32, K=400, K_Search=256
测试范围: 1M 到 10M (step=1M)
Range: 1%, 10%, 20%, 100%

测试方法: hnsw, serf

记录指标:
- 构建时间 (build_time)
- 索引大小 (size_mb)
- recall
- qps

特点:
- 多线程构建索引
- 多线程生成 groundtruth
- 多线程查询
- 一次执行一个任务（避免资源竞争）
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

# 可执行文件路径
SERF_BINARY = os.path.join(SERF_ROOT, "build/benchmark/serf_multithread")
HNSW_BINARY = os.path.join(SERF_ROOT, "build/benchmark/hnsw_multithread")

# 索引保存目录
INDEX_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability", "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability")

# 数据集路径 - 使用 DEEP 附带的 query 文件
DEEP_BASE_PATH = os.environ.get("GRAB_DEEP10M_BASE", os.environ.get("GRAB_DEEP_BASE", ""))
DEEP_QUERY_PATH = os.environ.get("GRAB_DEEP_QUERY", "")

# Leap strategy（SeRF专用）
STRATEGY = "MAX_POS"

# 固定参数
FIXED_M = 32
FIXED_K = 400
FIXED_K_SEARCH = 256

# 测试规模点：1M到10M，步长1M
DATA_SIZES = [1000000 * i for i in range(1, 11)]

# Range 百分比: 1%, 10%, 20%, 100%
RANGE_PCTS = [1.0, 10.0, 20.0, 100.0]

# 测试方法
METHODS = ["hnsw", "serf"]

# OpenMP线程数配置
OMP_THREADS_BUILD = 30      # 构建索引时使用的线程数
OMP_THREADS_QUERY = 30      # 查询时生成groundtruth使用的线程数

# 查询线程数（用于多线程查询）
NUM_QUERY_THREADS = 30

# 并行任务数 - 一次只执行一个任务，避免资源竞争
WORKERS = 1

# ============== 工具函数 ==============

def get_index_path(method, data_size, m, k):
    """生成索引文件路径"""
    size_suffix = f"{data_size//1000000}m"
    return os.path.join(INDEX_DIR, f"{method}_deep_{size_suffix}_M{m}_K{k}.bin")

def index_exists(method, data_size, m, k):
    """检查索引是否存在且非空"""
    path = get_index_path(method, data_size, m, k)
    return os.path.exists(path) and os.path.getsize(path) > 0

def get_index_size_mb(method, data_size, m, k):
    """获取索引文件大小（MB）"""
    path = get_index_path(method, data_size, m, k)
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return None

def get_binary(method):
    """获取对应方法的可执行文件路径"""
    return SERF_BINARY if method == "serf" else HNSW_BINARY

# ============== 任务生成 ==============

def generate_tasks():
    """生成所有测试任务（构建+查询）"""
    tasks = []

    for size in DATA_SIZES:
        for method in METHODS:
            # 检查索引是否已存在
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

# ============== 任务执行 ==============

def run_build(task):
    """执行构建任务"""
    method = task["method"]
    size = task["data_size"]
    m = task["M"]
    k = task["K"]

    index_path = task["index_path"]
    binary = get_binary(method)

    # 设置环境变量 - 多线程构建
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS_BUILD)

    # 构建命令
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

    # SeRF专用参数
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    size_str = f"{size//1000000}M"

    print(f"[BUILD] {method} deep_{size_str} M={m} K={k} ef_search={FIXED_K_SEARCH}")

    output_file = f"/tmp/{method}_deep_{size_str}_M{m}_K{k}_build.txt"

    try:
        os.makedirs(INDEX_DIR, exist_ok=True)

        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # 解析构建时间
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

        # 获取索引大小
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
    """执行查询任务"""
    method = task["method"]
    size = task["data_size"]
    m = task["M"]
    k = task["K"]

    index_path = task["index_path"]
    binary = get_binary(method)

    # 检查索引是否存在
    if not os.path.exists(index_path):
        print(f"[SKIP] {method} deep_{size//1000000}M: 索引不存在")
        return []

    # 设置环境变量 - 多线程生成groundtruth
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS_QUERY)

    # 构建命令
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
        "-threads", str(NUM_QUERY_THREADS),  # 多线程查询
    ]

    # SeRF专用参数
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    size_str = f"{size//1000000}M"

    print(f"[QUERY] {method} deep_{size_str} M={m} K={k} ef_search={FIXED_K_SEARCH}")

    output_file = f"/tmp/{method}_deep_{size_str}_M{m}_K{k}_query.txt"

    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # 解析结果
        results = []
        current_k_search = None

        with open(output_file, 'r') as f:
            for line in f:
                # 检测当前测试的K_Search值
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

                        # 只保存我们关心的 range_pct
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
    """执行单个任务的完整流程：构建（如需要）+ 查询"""
    method = task["method"]
    size = task["data_size"]
    size_str = f"{size//1000000}M"

    print(f"\n{'='*80}")
    print(f"[TASK] {method} deep_{size_str} M={FIXED_M} K={FIXED_K} K_Search={FIXED_K_SEARCH}")
    print(f"{'='*80}")

    results = []
    build_time = None
    index_size_mb = task.get("index_size_mb")

    # 如果索引不存在，先构建
    if not task["has_index"]:
        build_result = run_build(task)
        if build_result is None:
            print(f"[FAIL] 构建失败，跳过查询")
            return []

        build_time = build_result["build_time"]
        index_size_mb = build_result["index_size_mb"]
    else:
        print(f"[INFO] 索引已存在: {index_size_mb:.2f}MB")

    # 执行查询
    query_results = run_query(task)

    # 添加构建时间和索引大小到结果中
    for r in query_results:
        r["build_time"] = build_time if build_time is not None else 0
        r["size_mb"] = index_size_mb

    return query_results

# ============== 主函数 ==============

def main():
    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*80)
    print("DEEP10M 固定参数规模扩展测试")
    print("="*80)
    print(f"数据集: {DEEP_BASE_PATH}")
    print(f"Query:  {DEEP_QUERY_PATH}")
    print(f"规模点: {[f'{s//1000000}M' for s in DATA_SIZES]}")
    print(f"固定参数: M={FIXED_M}, K={FIXED_K}, K_Search={FIXED_K_SEARCH}")
    print(f"Range: {RANGE_PCTS}%")
    print(f"方法: {METHODS}")
    print("="*80)
    print(f"构建线程数: {OMP_THREADS_BUILD}")
    print(f"查询线程数: {NUM_QUERY_THREADS}")
    print(f"Groundtruth线程数: {OMP_THREADS_QUERY}")
    print(f"并行任务数: {WORKERS}")
    print("="*80)

    # 生成任务
    tasks = generate_tasks()

    print(f"\n共 {len(tasks)} 个任务")
    print(f"任务列表:")
    for task in tasks:
        status = "已有索引" if task["has_index"] else "需构建"
        print(f"  - {task['method']} deep_{task['data_size']//1000000}M: {status}")

    # 执行任务
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
            print(f"\n[PROGRESS] 进度: {completed}/{len(tasks)}")

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        result_path = os.path.join(RESULTS_DIR, f"fixed_params_m32_k400_ef256_{TIMESTAMP}.csv")
        df.to_csv(result_path, index=False)

        print("\n" + "="*80)
        print(f"结果已保存到: {result_path}")
        print(f"共 {len(all_results)} 条记录")
        print("="*80)

        # 显示统计信息
        print("\n各数据集各方法的记录数:")
        summary = df.groupby(['dataset', 'method', 'range_pct']).size().reset_index(name='count')
        print(summary.to_string(index=False))

        # 显示各方法各规模的构建时间、索引大小、recall、qps
        print("\n汇总结果 (range=10%):")
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
