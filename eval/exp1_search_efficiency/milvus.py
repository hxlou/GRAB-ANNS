#!/usr/bin/env python3
"""
Pareto Frontier Test Script for Milvus

目标：
- 测试 Milvus HNSW 索引在多个数据集上的性能
- 测试多个构建参数组合 (M, ef_construction)
- 测试多个搜索参数 (ef_search)
- 收集 QPS-Recall 数据用于绘制帕累托前沿曲线
- 输出与 HNSW/SeRF 一致的 CSV 格式

特点：
- 复用已有 collection（避免重复构建）
- 断点续传（检查已有结果）
- CPU 亲和性（绑定到 E-cores）
- 与 HNSW/SeRF 相同的 query 文件

输出：
- CSV 文件包含 method, dataset, M, K, K_Search, range_pct, recall, qps, comps, build_time, index_size_mb
- 与 HNSW/SeRF 共享同一个 CSV 文件：results/pareto_frontier/pareto_all.csv
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

# Milvus 连接
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# 输出目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp1_search_efficiency")

# CPU 亲和性 - 绑定到 E-cores（避免与 HNSW/SeRF 竞争）
E_CORES = list(range(16, 32))

def set_cpu_affinity():
    """绑定进程到 E-cores"""
    try:
        os.sched_setaffinity(0, E_CORES)
        print(f"[CPU] 绑定到 E-cores: {E_CORES}")
    except AttributeError:
        print("[WARN] os.sched_setaffinity 不可用")
    except Exception as e:
        print(f"[WARN] 设置 CPU 亲和性失败: {e}")

# 在导入时设置 CPU 亲和性
set_cpu_affinity()

# 数据集配置（与 HNSW/SeRF 一致）
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

# 数据规模
DATA_SIZE = 1000000

# 索引参数（与 HNSW/SeRF 一致）
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

# 搜索参数（与 HNSW/SeRF 一致）
EF_SEARCH_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Range 百分比
RANGE_PCTS = [1.0, 10.0, 20.0, 100.0]

# 查询配置
NUM_QUERIES = 1000
TOP_K = 10
QUERY_SEED = 42

# Milvus 索引配置
INDEX_TYPE = "HNSW"
METRIC_TYPE = "L2"

# ============== 工具函数 ==============

def load_fvecs(path, count=None):
    """加载 fvecs 文件"""
    with open(path, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if count is None:
            data = np.frombuffer(f.read(), dtype=np.float32)
        else:
            bytes_to_read = int(count) * int(dim) * 4
            data = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
        return data.reshape(-1, dim)

def get_batch_size(dim):
    """根据维度获取合适的批量大小（避免超过 gRPC 64MB 限制）"""
    if dim >= 2048:
        return 7500
    elif dim >= 960:
        return 16000
    elif dim >= 128:
        return 120000
    else:
        return 150000

def synthesize_queries(vectors, num_queries=NUM_QUERIES, seed=QUERY_SEED):
    """生成合成查询（与 C++ 一致的方法）"""
    n, dim = vectors.shape
    queries = np.zeros((num_queries, dim), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for i in range(num_queries):
        random_indices = rng.integers(0, n, size=dim)
        for j in range(dim):
            queries[i, j] = vectors[random_indices[j], j]

    return queries

def get_collection_name(dataset_key, m, ef_con):
    """生成 collection 名称"""
    dataset_short = dataset_key.split('-')[0].lower()
    return f"{dataset_short}_1m_M{m}_EFCON{ef_con}"

def get_existing_results(csv_path):
    """读取已测试的配置"""
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
        print(f"[WARN] 无法读取已有结果: {e}")
        return set()

# ============== Milvus 操作 ==============

def create_collection_if_needed(collection_name, dim, vectors, m, ef_con):
    """创建 collection 并插入数据（如果不存在）"""
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        num_entities = collection.num_entities

        if num_entities == DATA_SIZE:
            print(f"  [INFO] Collection 已存在: {collection_name} ({num_entities} vectors)")
            return collection
        else:
            print(f"  [WARN] Collection 数据不完整: {collection_name} ({num_entities}/{DATA_SIZE})")
            print(f"  [INFO] 删除并重建...")
            utility.drop_collection(collection_name)

    # 创建新 collection
    print(f"  [CREATE] 创建 collection: {collection_name}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description=f"M={m}, ef_con={ef_con}")
    collection = Collection(name=collection_name, schema=schema)

    # 插入数据
    actual_size = vectors.shape[0]
    ids = np.arange(actual_size)
    batch_size = get_batch_size(dim)

    print(f"  [INSERT] 插入 {actual_size} 向量...")
    for i in range(0, actual_size, batch_size):
        end = min(i + batch_size, actual_size)
        collection.insert([ids[i:end].tolist(), vectors[i:end]])
    collection.flush()
    print(f"  [INSERT] 完成: {actual_size} vectors")

    return collection

def build_index_if_needed(collection, m, ef_con):
    """构建索引（如果不存在）"""
    indexes = collection.indexes
    if indexes:
        print(f"  [INFO] 索引已存在")
        # 验证索引是否就绪
        try:
            collection.load()
            collection.release()
            return True, 0.0
        except Exception as e:
            print(f"  [WARN] 索引未就绪: {e}")
            return False, 0.0

    # 构建索引
    print(f"  [INDEX] 构建 HNSW: M={m}, efConstruction={ef_con}")
    print(f"  [INDEX] 预计需要 1-5 分钟...")

    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"M": m, "efConstruction": ef_con}
    }

    t0 = time.time()
    collection.create_index(field_name="embedding", index_params=index_params)
    build_time = time.time() - t0

    # 验证索引
    try:
        collection.load()
        collection.release()
        print(f"  [INDEX] ✅ 构建完成: {build_time:.2f}s")
        return True, build_time
    except Exception as e:
        print(f"  [ERROR] 索引构建失败: {e}")
        return False, build_time

def test_range_query_batch(collection, queries, vectors, ef, range_pct):
    """执行范围查询并计算 recall（批量处理）"""
    actual_size = vectors.shape[0]
    range_width = int(actual_size * range_pct / 100)
    range_width = max(1, range_width)

    # 生成随机范围
    rng = np.random.default_rng(QUERY_SEED)
    if range_pct == 100:
        l_bounds = np.zeros(NUM_QUERIES, dtype=int)
    else:
        max_l_bound = actual_size - range_width - 80
        max_l_bound = max(0, max_l_bound)
        l_bounds = rng.integers(0, max_l_bound + 1, size=NUM_QUERIES)

    # 计算 ground truth
    total_recall = 0.0
    total_time = 0.0

    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": ef}}

    collection.load()  # 确保已加载

    for query_idx in range(NUM_QUERIES):
        l_bound = int(l_bounds[query_idx])
        r_bound = l_bound + range_width - 1

        # Ground truth
        range_vectors = vectors[l_bound:r_bound+1]
        query_vector = queries[query_idx:query_idx+1]

        # 计算距离（内存优化）
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

        # Milvus 查询
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

        # 计算 recall
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

# ============== 主流程 ==============

def run_single_task(task, csv_path, existing_results):
    """执行单个任务"""
    dataset_key = task["dataset_key"]
    dataset_info = task["dataset_info"]
    m = task["M"]
    ef_con = task["K"]

    collection_name = get_collection_name(dataset_key, m, ef_con)

    print(f"\n{'='*80}")
    print(f"[TASK] {dataset_key} M={m} K={ef_con}")
    print(f"{'='*80}")

    # 过滤出未测试的 ef_search
    untested_ef = []
    for ef in EF_SEARCH_VALUES:
        key = ('milvus', dataset_key, m, ef_con, ef)
        if key not in existing_results:
            untested_ef.append(ef)

    if not untested_ef:
        print(f"[SKIP] 所有 ef_search 已测试")
        return []

    print(f"[INFO] 待测试 ef_search: {untested_ef}")

    # 加载数据
    print(f"[LOAD] 加载数据: {dataset_info['base_path']}")
    vectors = load_fvecs(dataset_info['base_path'], DATA_SIZE)
    print(f"[LOAD] 数据大小: {vectors.shape}")

    # 加载或生成查询
    query_path = dataset_info['query_path']
    if query_path and os.path.exists(query_path):
        queries = load_fvecs(query_path, NUM_QUERIES)
        print(f"[LOAD] 查询数据: {queries.shape}")
    else:
        print(f"[SYNTH] 生成合成查询: {NUM_QUERIES} 条")
        queries = synthesize_queries(vectors, NUM_QUERIES)

    # 创建/加载 collection
    collection = create_collection_if_needed(collection_name, dataset_info['dim'], vectors, m, ef_con)

    # 构建索引
    index_ready, build_time = build_index_if_needed(collection, m, ef_con)

    if not index_ready:
        print(f"[ERROR] 索引未就绪，跳过测试")
        return []

    # 计算索引大小（Milvus collection 的估算大小）
    # Milvus 的 collection 大小约等于 数据量 × 维度 × 4 bytes（float32）
    # 加上索引结构开销（约 1.5-2 倍）
    data_size_mb = (DATA_SIZE * dataset_info['dim'] * 4) / (1024 * 1024)
    index_size_mb = data_size_mb * 1.5  # 索引结构开销

    # 执行查询测试
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

    # 释放 collection
    collection.release()
    print(f"[INFO] Collection 已释放")

    return results

def generate_tasks():
    """生成所有任务"""
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
    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 使用统一 CSV 文件（与 HNSW/SeRF 共享）
    UNIFIED_CSV = os.path.join(RESULTS_DIR, "pareto_all.csv")
    CSV_PATH = UNIFIED_CSV

    print("="*80)
    print("Pareto Frontier Test for Milvus")
    print("="*80)
    print(f"数据集: {list(DATASETS.keys())}")
    print(f"数据规模: {DATA_SIZE//1000000}M")
    print(f"索引参数: {len(INDEX_CONFIGS)} 个组合")
    print(f"搜索参数: ef_search = {EF_SEARCH_VALUES}")
    print(f"Range: {RANGE_PCTS}%")
    print("="*80)

    # 连接 Milvus
    print("\n[CONNECT] 连接 Milvus...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=60)
        print(f"[CONNECT] ✅ 已连接: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"[ERROR] 连接失败: {e}")
        print(f"[HINT] 请先启动 Milvus: cd script_milvus && ./start_milvus.sh")
        return 1

    # 生成任务
    tasks = generate_tasks()
    print(f"\n共 {len(tasks)} 个任务")

    # 创建 CSV 文件（如果不存在）
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write("method,dataset,M,K,K_Search,range_pct,recall,qps,comps,build_time,index_size_mb\n")
        print(f"[INFO] 创建统一 CSV 文件: {CSV_PATH}")
    else:
        print(f"[INFO] 使用已有 CSV 文件: {CSV_PATH}")

    # 读取已有结果
    existing_results = get_existing_results(CSV_PATH)
    print(f"已测试配置: {len(existing_results)} 个")

    # 顺序执行任务
    completed = 0
    for task_idx, task in enumerate(tasks, 1):
        print(f"\n[PROGRESS] 任务 {task_idx}/{len(tasks)}")

        results = run_single_task(task, CSV_PATH, existing_results)

        # 实时写入 CSV
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

    return 0

if __name__ == "__main__":
    sys.exit(main())
