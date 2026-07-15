#!/usr/bin/env python3
"""
Experiment 1 (Q1) - Milvus build-time only

需求（按用户说明）：
- 仅测试构建（不做 recall/QPS）
- 固定参数：M=32, K=400（这里 K 记作 efConstruction=400），K_Search 留空
- 4 个数据集：DEEP-96, SIFT-128, GIST-960, WIT-2048
- 若 collection 已存在：先 drop，再 sleep 60s
- create_index 完成后：记录 build_time，然后 drop，再 sleep 60s
- 输出 CSV 列：method,dataset,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips

说明：
- build_time 计时口径：create_index + wait_for_index_building_complete（把异步构建等待算进去）
- ips 口径：DATA_SIZE / build_time（与用户提供的 SeRF/HNSW ips 计算方式一致）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

sys.stdout.reconfigure(line_buffering=True)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp4_ingestion")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============== DATASETS ==============

DATASETS = {
    "DEEP-96": {
        "base_path": os.environ.get(
            "DEEP_BASE_PATH",
            "",
        ),
        "dim": 96,
    },
    "SIFT-128": {
        "base_path": os.environ.get(
            "SIFT_BASE_PATH",
            "",
        ),
        "dim": 128,
    },
    "GIST-960": {
        "base_path": os.environ.get(
            "GIST_BASE_PATH",
            "",
        ),
        "dim": 960,
    },
    "WIT-2048": {
        "base_path": os.environ.get(
            "WIT_BASE_PATH",
            "",
        ),
        "dim": 2048,
    },
}

# ============== MILVUS CONFIG ==============

MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")

METHOD = "milvus"
M = 32
K = 400  # 写入 CSV 的 K 列；对应 Milvus HNSW efConstruction
EF_CONSTRUCTION = 400

DEFAULT_DATA_SIZE = 1_000_000
DEFAULT_SLEEP_SEC = int(os.environ.get("SLEEP_SEC", "60"))
DEFAULT_INDEX_WAIT_TIMEOUT_SEC = int(os.environ.get("INDEX_WAIT_TIMEOUT_SEC", "7200"))


def get_insert_batch_size(dim: int) -> int:
    if dim >= 2048:
        return 7500
    if dim >= 960:
        return 16000
    if dim >= 128:
        return 120000
    return 150000


def ensure_milvus_connection(max_retries: int = 10, sleep_sec: int = 3) -> None:
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            connections.connect(
                "default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=30,
                overwrite=True,
            )
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}: {last_err}")


def milvus_call(op_name: str, func, *args, **kwargs):
    retries = 3
    sleep_sec = 10
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                print(f"[WARN] {op_name} failed, retry {attempt+1}/{retries}: {e}")
                try:
                    ensure_milvus_connection(max_retries=3, sleep_sec=2)
                except Exception:
                    pass
                time.sleep(sleep_sec)
            else:
                raise
    raise RuntimeError(f"{op_name} failed after retries: {last_err}")


def get_fvecs_dim(path: str) -> int:
    with open(path, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)
        if dim.size != 1:
            raise RuntimeError(f"Failed to read dim from fvecs: {path}")
        return int(dim[0])


def fvecs_memmap(path: str, dim: int) -> np.ndarray:
    mm = np.memmap(path, dtype=np.int32, mode="r")
    stride = dim + 1
    if mm.size % stride != 0:
        raise RuntimeError(
            f"Invalid fvecs layout: total_ints={mm.size} not divisible by (dim+1)={stride} for {path}"
        )
    total = mm.size // stride
    m2 = mm.reshape(total, stride)
    sample = m2[: min(1024, total), 0]
    if np.any(sample != dim):
        raise RuntimeError(
            f"Unexpected fvecs dim prefix (expected {dim}) in {path}: sample={np.unique(sample)[:10]}"
        )
    return m2[:, 1:].view(np.float32)


def collection_name(dataset: str, data_size: int) -> str:
    dataset_short = dataset.split("-")[0].lower()
    return f"exp1_buildonly_{dataset_short}_{data_size//1_000_000}m_M{M}_K{K}"


@dataclass(frozen=True)
class Row:
    method: str
    dataset: str
    M: int
    K: int
    K_Search: str
    range_pct: str
    recall: str
    qps: str
    comps: str
    build_time: float
    ips: float


def build_one_dataset(
    *,
    dataset: str,
    base_path: str,
    dim: int,
    data_size: int,
    sleep_sec: int,
    index_wait_timeout_sec: int,
) -> Row:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"[{dataset}] base_path not found: {base_path}")
    file_dim = get_fvecs_dim(base_path)
    if file_dim != dim:
        raise RuntimeError(f"[{dataset}] dim mismatch: file={file_dim}, expected={dim}")

    print(f"\n{'='*80}\n[DATASET] {dataset} (dim={dim}, N={data_size:,})\n{'='*80}")
    vectors = fvecs_memmap(base_path, dim)
    actual_size = min(data_size, vectors.shape[0])

    ensure_milvus_connection()
    col_name = collection_name(dataset, data_size)

    if utility.has_collection(col_name):
        print(f"[CLEANUP] drop existing collection: {col_name}")
        milvus_call("drop_collection", utility.drop_collection, col_name)
        if sleep_sec > 0:
            print(f"[SLEEP] {sleep_sec}s (after initial drop)")
            time.sleep(sleep_sec)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description=f"exp1 build-only {dataset} M={M} efc={EF_CONSTRUCTION}")
    collection = Collection(name=col_name, schema=schema)

    batch_size = get_insert_batch_size(dim)
    print(f"[INSERT] inserting {actual_size:,} vectors (batch={batch_size})")
    for start in range(0, actual_size, batch_size):
        end = min(start + batch_size, actual_size)
        ids = np.arange(start, end, dtype=np.int64)
        emb = np.asarray(vectors[start:end], dtype=np.float32).copy()
        milvus_call("insert", collection.insert, [ids.tolist(), emb])
        if (start // batch_size) % 20 == 0:
            print(f"  progress: {end:,}/{actual_size:,} ({end/actual_size*100:.1f}%)")

    milvus_call("flush", collection.flush)

    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": M, "efConstruction": EF_CONSTRUCTION},
    }

    print(f"[INDEX] create_index + wait_complete (M={M}, efConstruction={EF_CONSTRUCTION})")
    t0 = time.time()
    milvus_call("create_index", collection.create_index, field_name="embedding", index_params=index_params)
    milvus_call(
        "wait_for_index_building_complete",
        utility.wait_for_index_building_complete,
        col_name,
        "",
        index_wait_timeout_sec,
    )
    build_time = time.time() - t0
    ips = float(actual_size) / build_time if build_time > 0 else float("nan")
    print(f"[INDEX] done: {build_time:.6f}s, ips={ips:.6f}")

    print(f"[CLEANUP] drop collection: {col_name}")
    milvus_call("drop_collection", utility.drop_collection, col_name)
    connections.disconnect("default")

    if sleep_sec > 0:
        print(f"[SLEEP] {sleep_sec}s (after final drop)")
        time.sleep(sleep_sec)

    return Row(
        method=METHOD,
        dataset=dataset,
        M=M,
        K=K,
        K_Search="",
        range_pct="",
        recall="",
        qps="",
        comps="",
        build_time=build_time,
        ips=ips,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size", type=int, default=DEFAULT_DATA_SIZE)
    parser.add_argument("--sleep-sec", type=int, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--index-wait-timeout-sec", type=int, default=DEFAULT_INDEX_WAIT_TIMEOUT_SEC)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(RESULTS_DIR, f"exp1_buildtime_only_milvus_M{M}_K{K}_{ts}.csv")

    rows = []
    for dataset, cfg in DATASETS.items():
        rows.append(
            build_one_dataset(
                dataset=dataset,
                base_path=cfg["base_path"],
                dim=cfg["dim"],
                data_size=args.data_size,
                sleep_sec=args.sleep_sec,
                index_wait_timeout_sec=args.index_wait_timeout_sec,
            ).__dict__
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[CSV] appended -> {out_path}")

    print(f"[DONE] results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
