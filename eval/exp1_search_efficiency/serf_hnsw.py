#!/usr/bin/env python3
"""
Pareto Frontier Test Script for SeRF/HNSW

目标：
- 测试多个构建参数组合 (M, ef_construction)
- 测试多个搜索参数 (ef_search)
- 收集 QPS-Recall 数据用于绘制帕累托前沿曲线
- 四个数据集：DEEP-96, SIFT-128, GIST-960, WIT-2048

特点：
- 多线程构建索引
- 多线程生成 groundtruth
- 多线程查询
- 顺序执行任务（避免资源竞争）
- 断点续传（检查已有索引和结果）

输出：
- CSV 文件包含 method, dataset, M, K, K_Search, range_pct, recall, qps, comps
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

# 可执行文件路径
SERF_BINARY = os.path.join(SERF_ROOT, "build/benchmark/serf_multithread")
HNSW_BINARY = os.path.join(SERF_ROOT, "build/benchmark/hnsw_multithread")

# 索引保存目录
INDEX_DIR = os.path.join(PROJECT_ROOT, "results", "exp1_search_efficiency", "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp1_search_efficiency")

# Leap strategy（SeRF专用）
STRATEGY = "MAX_POS"

# 数据集配置
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

# 数据规模（统一使用 1M）
DATA_SIZE = 1000000

# 测试方法
METHODS = ["hnsw", "serf"]

# 构建参数配置（多个组合用于探索帕累托前沿）
INDEX_CONFIGS = [
    # 低延迟配置
    {"M": 8,  "ef_con": 100},
    {"M": 8,  "ef_con": 200},

    # 通用配置
    {"M": 16, "ef_con": 200},
    {"M": 16, "ef_con": 400},

    # 高质量配置
    {"M": 32, "ef_con": 400},
    {"M": 32, "ef_con": 800},

    # 极致精度配置
    {"M": 64, "ef_con": 400},
    {"M": 64, "ef_con": 800},
]

# 搜索参数配置（多个 ef_search 值用于探索 recall-QPS 权衡）
EF_SEARCH_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

# Range 百分比（测试多个 range）
RANGE_PCTS = [1.0, 10.0, 20.0, 100.0]

# OpenMP 线程数配置
OMP_THREADS_BUILD = 30      # 构建索引时使用的线程数
OMP_THREADS_QUERY = 30      # 查询时生成groundtruth使用的线程数

# 查询线程数（用于多线程查询）
NUM_QUERY_THREADS = 30

# ============== 工具函数 ==============

def get_index_path(method, dataset_key, m, k):
    """生成索引文件路径"""
    dataset_short = dataset_key.split('-')[0].lower()  # DEEP-96 -> deep
    return os.path.join(INDEX_DIR, f"{method}_{dataset_short}_1m_M{m}_K{k}.bin")

def index_exists(method, dataset_key, m, k):
    """检查索引是否存在且非空"""
    path = get_index_path(method, dataset_key, m, k)
    return os.path.exists(path) and os.path.getsize(path) > 0

def get_index_size_mb(method, dataset_key, m, k):
    """获取索引文件大小（MB）"""
    path = get_index_path(method, dataset_key, m, k)
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return None

def get_binary(method):
    """获取对应方法的可执行文件路径"""
    return SERF_BINARY if method == "serf" else HNSW_BINARY

def get_existing_results(csv_path):
    """读取已有的测试结果，返回已测试的配置集合"""
    if not os.path.exists(csv_path):
        return set()

    try:
        df = pd.read_csv(csv_path)
        # 返回 (method, dataset, M, K, K_Search) 的集合
        existing = set()
        for _, row in df.iterrows():
            key = (row['method'], row['dataset'], row['M'], row['K'], row['K_Search'])
            existing.add(key)
        return existing
    except Exception as e:
        print(f"[WARN] 无法读取已有结果 {csv_path}: {e}")
        return set()

# ============== 任务生成 ==============

def generate_tasks(csv_path):
    """生成所有测试任务（构建+查询）"""
    tasks = []
    existing_results = get_existing_results(csv_path)

    for dataset_key, dataset_info in DATASETS.items():
        for method in METHODS:
            for config in INDEX_CONFIGS:
                m = config["M"]
                k = config["ef_con"]

                # 检查索引是否已存在
                has_index = index_exists(method, dataset_key, m, k)
                index_size = get_index_size_mb(method, dataset_key, m, k) if has_index else None

                # 检查是否所有 K_Search 都已测试
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

# ============== 任务执行 ==============

def run_build(task):
    """执行构建任务"""
    method = task["method"]
    dataset_key = task["dataset_key"]
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
        "-dataset", task["dataset_name"],
        "-N", str(DATA_SIZE),
        "-dataset_path", task["base_path"],
        "-query_path", task["query_path"],
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search_list", ",".join(map(str, EF_SEARCH_VALUES)),
        "-save_index", index_path,
        "-threads", str(NUM_QUERY_THREADS),  # 多线程查询
    ]

    # SeRF专用参数
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

            # 实时打印输出
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        build_time = time.time() - t0

        # 获取索引大小
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
    """执行查询任务（只测试未测试的 K_Search 值）"""
    method = task["method"]
    dataset_key = task["dataset_key"]
    m = task["M"]
    k = task["K"]

    # 如果没有传入构建信息，从 task 中获取
    if build_time is None:
        build_time = 0.0
    if index_size_mb is None:
        index_size_mb = task.get("index_size_mb", 0.0)

    index_path = task["index_path"]
    binary = get_binary(method)

    # 检查索引是否存在
    if not os.path.exists(index_path):
        print(f"[SKIP] {method} {dataset_key}: 索引不存在")
        return []

    # 过滤出未测试的 K_Search 值
    untested_k_search = []
    for k_search in EF_SEARCH_VALUES:
        key = (method, dataset_key, m, k, k_search)
        if key not in existing_results:
            untested_k_search.append(k_search)

    if not untested_k_search:
        print(f"[SKIP] {method} {dataset_key} M={m} K={k}: 所有 K_Search 已测试")
        return []

    # 设置环境变量 - 多线程生成groundtruth
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS_QUERY)

    # 构建命令（只测试未测试的 K_Search）
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
        "-threads", str(NUM_QUERY_THREADS),  # 多线程查询
    ]

    # SeRF专用参数
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    print(f"\n[QUERY] {method} {dataset_key} M={m} K={k} ef_search={untested_k_search}")

    output_file = f"/tmp/{method}_{dataset_key}_M{m}_K{k}_query.txt"

    try:
        t0 = time.time()
        with open(output_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      env=env, universal_newlines=True, bufsize=1)

            # 实时打印输出
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        query_time = time.time() - t0

        # 解析结果
        results = []
        current_k_search = None

        with open(output_file, 'r') as f:
            for line in f:
                # 检测当前测试的K_Search值
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

                        # 只保存我们关心的 range_pct
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
    """执行单个任务的完整流程：构建（如需要）+ 查询"""
    method = task["method"]
    dataset_key = task["dataset_key"]
    m = task["M"]
    k = task["K"]

    print(f"\n{'='*80}")
    print(f"[TASK] {method} {dataset_key} M={m} K={k}")
    print(f"{'='*80}")

    # 重新读取已有结果（支持断点续传）
    existing_results = get_existing_results(csv_path)

    # 获取构建信息
    build_time = 0.0
    index_size_mb = task.get("index_size_mb", 0.0)

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

    # 执行查询（只测试未测试的配置）
    query_results = run_query(task, existing_results, build_time, index_size_mb)

    return query_results

# ============== 主函数 ==============

def main():
    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 使用统一 CSV 文件
    UNIFIED_CSV = os.path.join(RESULTS_DIR, "pareto_all.csv")
    CSV_PATH = UNIFIED_CSV

    print("="*80)
    print("Pareto Frontier Test for SeRF/HNSW")
    print("="*80)
    print(f"数据集: {list(DATASETS.keys())}")
    print(f"数据规模: {DATA_SIZE//1000000}M")
    print(f"方法: {METHODS}")
    print(f"构建参数: {len(INDEX_CONFIGS)} 个组合")
    print(f"搜索参数: ef_search = {EF_SEARCH_VALUES}")
    print(f"Range: {RANGE_PCTS}%")
    print("="*80)
    print(f"构建线程数: {OMP_THREADS_BUILD}")
    print(f"查询线程数: {NUM_QUERY_THREADS}")
    print(f"Groundtruth线程数: {OMP_THREADS_QUERY}")
    print("="*80)

    # 生成任务
    tasks = generate_tasks(CSV_PATH)

    print(f"\n共 {len(tasks)} 个任务")

    # 统计任务状态
    tasks_with_index = sum(1 for t in tasks if t["has_index"])
    tasks_all_tested = sum(1 for t in tasks if t["all_tested"])
    tasks_need_build = sum(1 for t in tasks if not t["has_index"])
    tasks_need_query = sum(1 for t in tasks if not t["all_tested"])

    print(f"  - 已有索引: {tasks_with_index}/{len(tasks)}")
    print(f"  - 需要构建: {tasks_need_build}/{len(tasks)}")
    print(f"  - 已完成测试: {tasks_all_tested}/{len(tasks)}")
    print(f"  - 需要查询: {tasks_need_query}/{len(tasks)}")

    # 创建 CSV 文件（如果不存在）
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write("method,dataset,M,K,K_Search,range_pct,recall,qps,comps,build_time,index_size_mb\n")
        print(f"[INFO] 创建统一 CSV 文件: {CSV_PATH}")
    else:
        print(f"[INFO] 使用已有 CSV 文件: {CSV_PATH}")

    # 顺序执行任务
    completed = 0
    for task_idx, task in enumerate(tasks, 1):
        print(f"\n[PROGRESS] 任务 {task_idx}/{len(tasks)}")

        results = run_single_task(task, CSV_PATH)

        # 实时写入统一 CSV
        if results:
            with open(CSV_PATH, 'a') as f:
                for r in results:
                    row = f"{r['method']},{r['dataset']},{r['M']},{r['K']},{r['K_Search']},{r['range_pct']},{r['recall']},{r['qps']},{r['comps']},{r['build_time']},{r['index_size_mb']}\n"
                    f.write(row)
            completed += len(results)

    print("\n" + "="*80)
    print(f"测试完成！")
    print(f"结果已保存到: {CSV_PATH}")
    print(f"共 {completed} 条新记录")
    print("="*80)

    # 显示汇总统计
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print("\n各数据集各方法的记录数:")
        summary = df.groupby(['dataset', 'method']).size().reset_index(name='count')
        print(summary.to_string(index=False))

    return 0

if __name__ == "__main__":
    sys.exit(main())
