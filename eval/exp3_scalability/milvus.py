#!/usr/bin/env python3
"""
DEEP10M Milvus 规模扩展测试脚本

索引参数组合: 多组 (M, ef_construction)
搜索参数组合: 多组 ef (K_Search)
测试范围: 1M 到 10M (step=1M)
Range: 1%, 10%, 20%, 100%

记录指标:
- 构建时间 (build_time)
- 索引大小 (size_mb) - 需要特殊处理（Milvus存储在Docker内）
- recall
- qps

特点:
- 增量保存结果（防止中断丢失数据）
- 进度提示（解决"卡住"焦虑）
- CPU affinity 绑定到 E-cores
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

# 禁用输出缓冲
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
# 注意：GT 生成时会解除绑定，使用所有核心
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
    """释放 CPU affinity，允许使用所有核心"""
    try:
        all_cores = list(range(cpu_count()))
        os.sched_setaffinity(0, all_cores)
        print(f"[CPU Affinity] Released to all cores: {all_cores}")
    except AttributeError:
        print("[WARNING] os.sched_setaffinity not available")
    except Exception as e:
        print(f"[WARNING] Failed to release CPU affinity: {e}")

def get_current_affinity():
    """获取当前 CPU 亲和性"""
    try:
        return os.sched_getaffinity(0)
    except AttributeError:
        return E_CORES
    except Exception as e:
        return E_CORES

# Milvus 连接
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

# 输出目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability")
os.makedirs(RESULTS_DIR, exist_ok=True)

# GT 缓存目录
GT_CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "exp3_scalability", "gt_cache")
os.makedirs(GT_CACHE_DIR, exist_ok=True)

# 数据集路径 - 使用 DEEP 附带的 query 文件
DEEP_BASE_PATH = os.environ.get("GRAB_DEEP10M_BASE", os.environ.get("GRAB_DEEP_BASE", ""))
DEEP_QUERY_PATH = os.environ.get("GRAB_DEEP_QUERY", "")

# 索引参数组合（与 deep10m_scaling 对齐）
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

# 搜索参数组合（与 deep10m_scaling 对齐）
# 注意：ef 必须 >= TOP_K，所以从 16 开始（TOP_K=10）
K_SEARCH_VALUES = [16, 32, 64, 128, 256, 512, 1024]

# 测试规模点：1M到10M，步长1M
DATA_SIZES = [1000000 * i for i in range(10, 11)]

# Range 百分比: 1%, 10%, 20%, 100%
RANGE_PCTS = [10.0]

# CSV 输出列顺序（与 deep10m_scaling 对齐）
CSV_COLUMNS = [
    "method", "dataset", "M", "K", "K_Search", "range_pct",
    "recall", "qps", "comps", "build_time", "size_mb",
]

# 查询配置
NUM_QUERIES = 1000  # 使用 1000 个查询进行批量测试
TOP_K = 10
METRIC_TYPE = "L2"
DIM = 96  # DEEP 数据集维度

# Batch insert size
BATCH_SIZE = 1000

# Batch query size - 将所有查询打包成 1批发送给 Milvus
BATCH_QUERY_SIZE = 200  # 200 queries per batch

# ============== GT CACHE FUNCTIONS ==============

# 全局变量（用于 worker 进程共享数据）
_shared_vectors = None
_shared_queries = None

def get_gt_cache_path(size):
    """获取 GT 缓存文件路径"""
    size_str = f"{size//1000000}m"
    return os.path.join(GT_CACHE_DIR, f"deep_{size_str}_gt_cache.pkl")

def _compute_single_query_gt(args):
    """计算单个查询的 GT（轻量级参数版本，使用共享数据）"""
    query_idx, l_bound, r_bound, top_k = args
    # 使用全局共享的 vectors 和 queries
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
    """预计算 Ground Truth 并缓存到文件（多线程版本）

    GT 结构: {range_pct: [(l_bound, r_bound, [gt_ids_for_each_query])]}
    每个 range_pct 生成 num_ranges_per_pct 个随机 range

    并行策略：每个查询独立计算，充分利用所有 CPU 核心
    """
    cache_path = get_gt_cache_path(size)

    # 检查缓存是否存在
    if os.path.exists(cache_path):
        if verbose:
            print(f"[GT CACHE] Found existing cache: {cache_path}")
            print(f"[GT CACHE] Loading precomputed GT...")
        with open(cache_path, 'rb') as f:
            gt_cache = pickle.load(f)
        if verbose:
            print(f"[GT CACHE] Loaded {len(gt_cache)} range_pct values from cache")
        return gt_cache

    # 缓存不存在，需要生成
    # 使用多进程池 - 优化后可以安全使用 30 个 worker
    num_workers = 30  # fork + 全局只读数组，避免每进程拷贝

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
                # 生成随机 range
                range_width = max(1, int(actual_size * range_pct / 100))
                max_l_bound = actual_size - range_width
                l_bound = np.random.randint(0, max(1, max_l_bound))
                r_bound = l_bound + range_width - 1

                # 准备所有查询的任务参数 - 1000 个任务（轻量级参数，只传索引）
                query_tasks = [(query_idx, l_bound, r_bound, TOP_K) for query_idx in range(num_queries)]

                # 多进程并行计算 1000 个查询的 GT（共享内存 + fork 避免拷贝）
                results = pool.map(_compute_single_query_gt, query_tasks)

                # 按查询索引排序，确保顺序正确
                results.sort(key=lambda x: x[0])
                all_gt_ids = [gt_ids for _, gt_ids in results]

                range_list.append((l_bound, r_bound, all_gt_ids))

            gt_cache[range_pct] = range_list

            if verbose:
                print(f"Done ({len(range_list)} ranges, {num_queries} queries each)")

    _shared_vectors = None
    _shared_queries = None

    # 保存到文件
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
    """生成 collection 名称"""
    size_str = f"{size//1000000}m"
    return f"deep_{size_str}_M{m}_EFCON{ef_con}"

def collection_has_index(collection):
    """检查 collection 是否有索引"""
    indexes = collection.indexes
    return len(indexes) > 0

# ============== INDEX SIZE ESTIMATION ==============

def estimate_index_size(size, m, ef_con):
    """
    估算 HNSW 索引大小

    基于 HNSW 的内存占用公式：
    - 每个节点约: M * 2 * (4 + 4) bytes = M * 16 bytes (边)
    - 每个节点: 4 bytes (level)
    - 每个向量: dim * 4 bytes
    - 每个ID: 8 bytes

    参考：https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    """
    # 每个节点的边数（约为 M * 2）
    edges_per_node = m * 2

    # 每条边占用的字节（neighbor_id + link_data）
    bytes_per_edge = 4 + 4

    # 每个节点的字节
    bytes_per_node = (
        edges_per_node * bytes_per_edge +  # edges
        4 +                                  # level
        DIM * 4 +                            # vector
        8                                    # id
    )

    total_bytes = size * bytes_per_node
    return total_bytes / (1024 * 1024)  # 转换为 MB

# ============== BUILD FUNCTION ==============

def build_collection(size, m, ef_con, verbose=True):
    """构建 collection 和索引

    注意：build_time 只计算 create_index() 的时间，不包括 insert 时间
    因为在 Milvus 中，insert 只是数据写入，create_index 才是真正的索引构建
    """
    if not ensure_milvus_connection():
        raise RuntimeError("Milvus connection unavailable during build.")
    collection_name = get_collection_name(size, m, ef_con)

    if verbose:
        print(f"\n{'='*80}")
        print(f"[BUILD] Milvus deep_{size//1000000}M M={m} ef_con={ef_con}")
        print(f"{'='*80}")
        print(f"  Collection: {collection_name}")

    # 强制删除现存 collection，确保每次都是全新构建（准确测量 build 时间）
    if utility.has_collection(collection_name):
        if verbose:
            print(f"  [CLEANUP] Dropping existing collection to ensure fresh build...")
        utility.drop_collection(collection_name)
        if verbose:
            print(f"  [CLEANUP] Dropped existing collection")

    # 创建 collection
    if verbose:
        print(f"  [CREATE] Creating new collection...")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields, description=f"DEEP-{size//1000000}M M={m} ef_con={ef_con}")
    collection = Collection(name=collection_name, schema=schema)

    # 插入数据（这不是索引构建！）
    if verbose:
        print(f"  [INSERT] Loading and inserting {size} vectors...")
        print(f"    注意：insert 只是数据写入，不计入 build_time")

    vectors = load_fvecs(DEEP_BASE_PATH, size)
    actual_size = vectors.shape[0]
    ids = np.arange(actual_size)

    insert_start = time.time()
    for i in range(0, actual_size, BATCH_SIZE):
        end = min(i + BATCH_SIZE, actual_size)
        milvus_call("insert", collection.insert, [ids[i:end].tolist(), vectors[i:end]])

        # 显示进度
        if verbose and (i // BATCH_SIZE) % 10 == 0:
            progress = (end / actual_size) * 100
            print(f"    Insert progress: {progress:.1f}% ({end}/{actual_size})")

    collection.flush()
    insert_time = time.time() - insert_start

    if verbose:
        print(f"  [INSERT] Inserted {actual_size} vectors in {insert_time:.2f}s (不计入 build_time)")

    # 构建索引 - 这才是真正的索引构建！只计算这部分时间
    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": "HNSW",
        "params": {"M": m, "efConstruction": ef_con}
    }

    if verbose:
        print(f"  [INDEX] Building HNSW index (真正的索引构建开始)...")
        print(f"    预计需要 5-15 分钟，请耐心等待...")
        print(f"    数据规模: {size:,} vectors")
        print(f"    参数: M={m}, efConstruction={ef_con}")

    build_start = time.time()
    milvus_call("create_index", collection.create_index, field_name="embedding", index_params=index_params)
    build_time = time.time() - build_start

    # 估算索引大小
    index_size_mb = estimate_index_size(size, m, ef_con)

    if verbose:
        print(f"  [INDEX] Index built in {build_time:.2f}s ({build_time/60:.1f} min)")
        print(f"  [SIZE]  Estimated index size: {index_size_mb:.2f} MB")
        print(f"  [TIMING] Insert: {insert_time:.2f}s, Build: {build_time:.2f}s (只有 Build 计入 build_time)")

    # 加载到内存
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
    """执行批量查询测试（使用缓存的 GT）

    Args:
        collection_name: Milvus collection 名称
        ef: 搜索参数 ef
        range_pcts: 要测试的 range 百分比列表
        gt_cache: 预计算的 ground truth 缓存
        queries: 查询向量（已加载，避免重复加载）
    """
    if not ensure_milvus_connection():
        raise RuntimeError("Milvus connection unavailable during query.")
    collection = Collection(collection_name)

    if verbose:
        print(f"\n[QUERY] Testing ef={ef}, ranges={range_pcts}")
        print(f"  [BATCH] 批量查询模式：{BATCH_QUERY_SIZE} 个查询/批")
        print(f"  [GT CACHE] 使用预计算的 GT，不再重复计算")

    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": ef}}

    results = []
    total_tests = len(range_pcts)
    current_test = 0

    for range_pct in range_pcts:
        current_test += 1

        if verbose:
            print(f"  [{current_test}/{total_tests}] Range={range_pct}%", end=" ", flush=True)

        # 从缓存获取该 range_pct 的所有 range 数据
        range_list = gt_cache[range_pct]

        total_recall = 0.0
        total_qps = 0.0
        valid_range_count = 0

        for range_idx, (l_bound, r_bound, all_gt_ids) in enumerate(range_list):
            expr = f"id >= {l_bound} && id <= {r_bound}"

            # Warm up（批量查询）
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

            # 批量查询：按 BATCH_QUERY_SIZE 分批
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

            # 计算 QPS（该 range 的总时间）
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
    """执行单个任务：构建（如需要）+ 多组查询

    Args:
        size: 数据集规模
        m, ef_con: 索引参数
        ef_list: 搜索参数列表
        gt_cache: 预计算的 ground truth 缓存
        queries: 查询向量（已加载，避免重复加载）
    """
    size_str = f"{size//1000000}M"
    collection_name = get_collection_name(size, m, ef_con)

    print(f"\n{'='*80}")
    print(f"[TASK] Milvus deep_{size_str} M={m} ef_con={ef_con} ef_list={ef_list}")
    print(f"{'='*80}")

    # 构建阶段
    build_result = build_collection(size, m, ef_con, verbose=True)

    results = []
    for ef in ef_list:
        # 查询阶段（使用缓存的 GT）
        query_results = query_collection(
            collection_name=build_result["collection_name"],
            ef=ef,
            range_pcts=RANGE_PCTS,
            gt_cache=gt_cache,
            queries=queries,
            verbose=True
        )

        # 组装结果
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

    # 释放内存（保留索引）
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
    print("DEEP10M Milvus 固定参数规模扩展测试")
    print("="*80)
    print(f"数据集: {DEEP_BASE_PATH}")
    print(f"Query:  {DEEP_QUERY_PATH}")
    print(f"规模点: {[f'{s//1000000}M' for s in DATA_SIZES]}")
    print(f"索引参数: {INDEX_CONFIGS}")
    print(f"搜索参数(ef): {K_SEARCH_VALUES}")
    print(f"Range: {RANGE_PCTS}%")
    print("="*80)

    # 连接 Milvus
    print("\n[CONNECT] Connecting to Milvus...")
    if not ensure_milvus_connection():
        return 1
    print(f"[CONNECT] Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    # 列出已有 collections
    all_collections = utility.list_collections()
    print(f"\n[INFO] Existing collections: {len(all_collections)}")
    for coll in all_collections:
        if coll.startswith("deep_"):
            print(f"  - {coll}")

    # 初始化 CSV（增量保存）
    csv_header = ",".join(CSV_COLUMNS)
    with open(result_path, 'w') as f:
        f.write(csv_header + '\n')

    # 加载查询向量（所有任务共用）
    print("\n[QUERIES] Loading query vectors...")
    queries = load_fvecs(DEEP_QUERY_PATH, NUM_QUERIES)
    print(f"[QUERIES] Loaded {queries.shape[0]} queries")

    # 执行任务（按数据集规模分组）
    total_tasks = len(DATA_SIZES) * len(INDEX_CONFIGS)
    all_results = []
    first_save = False

    task_num = 0
    for size in DATA_SIZES:
        size_str = f"{size//1000000}M"

        # 为当前数据集规模预计算 GT
        print(f"\n\n{'='*80}")
        print(f"[GT PRECOMPUTE] 预计算 {size_str} 的 Ground Truth")
        print(f"{'='*80}")

        # 动态调整 CPU 亲和性：GT 生成时使用全部核心
        original_affinity = get_current_affinity()
        print(f"[CPU Affinity] 保存当前亲和性: {original_affinity}")
        print(f"[CPU Affinity] 切换到全部核心用于 GT 生成...")
        release_cpu_affinity()

        try:
            gt_cache = precompute_ground_truth(size, RANGE_PCTS, num_queries=NUM_QUERIES, verbose=True)
        finally:
            # 恢复原始亲和性（E-cores）
            print(f"[CPU Affinity] 恢复到 E-cores: {E_CORES}")
            set_cpu_affinity()

        # 当前规模的所有索引参数测试（复用同一个 GT）
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

                    # 增量保存
                    df_new = pd.DataFrame(results, columns=CSV_COLUMNS)
                    df_new.to_csv(result_path, mode='a', header=first_save, index=False)

                    print(f"\n[PROGRESS] 进度: {task_num}/{total_tasks} | 已保存: {len(all_results)} 条记录")

            except Exception as e:
                print(f"\n[ERROR] Task failed: {e}")
                import traceback
                traceback.print_exc()

    # 最终汇总
    if all_results:
        df = pd.DataFrame(all_results)

        print("\n" + "="*80)
        print("测试完成!")
        print("="*80)
        print(f"结果文件: {result_path}")
        print(f"共 {len(all_results)} 条记录")

        # 显示汇总
        print("\n汇总结果 (range=10%):")
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
