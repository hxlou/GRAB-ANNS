#!/usr/bin/env python3
"""
Build-time benchmark for 4 datasets: SeRF vs filtered-HNSW (baseline HNSW with range filter).

This script runs the C++ binaries in *build-only* mode:
  - No groundtruth generation/loading
  - No query execution
  - Only measures `Build Index Time` printed by the binaries

Defaults:
  - M = 32  (index_k)
  - K = 400 (ef_construction)
  - 4 datasets: DEEP / SIFT / GIST / WIT

CSV format:
method,dataset,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    dim: int
    dataset_arg: str
    base_path: str


DATASETS = [
    Dataset(
        name="DEEP",
        dim=96,
        dataset_arg="deep",
        base_path=os.environ.get("GRAB_DEEP_BASE", ""),
    ),
    Dataset(
        name="SIFT",
        dim=128,
        dataset_arg="local",
        base_path=os.environ.get("GRAB_SIFT_BASE", ""),
    ),
    Dataset(
        name="GIST",
        dim=960,
        dataset_arg="local",
        base_path=os.environ.get("GRAB_GIST_BASE", ""),
    ),
    Dataset(
        name="WIT",
        dim=2048,
        dataset_arg="local",
        base_path=os.environ.get("GRAB_WIT_BASE", ""),
    ),
]


BUILD_TIME_RE = re.compile(r"Build Index Time:\s*([0-9]+(?:\.[0-9]+)?)s")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_binary_paths() -> tuple[str, str]:
    root = repo_root()
    return (
        str(root / "third_party" / "SeRF" / "build" / "benchmark" / "serf_multithread"),
        str(root / "third_party" / "SeRF" / "build" / "benchmark" / "hnsw_multithread"),
    )


def parse_build_time_s(output: str) -> float | None:
    m = BUILD_TIME_RE.search(output)
    if not m:
        return None
    return float(m.group(1))


def run_build_only(
    *,
    binary: str,
    dataset: Dataset,
    n_points: int,
    index_k: int,
    ef_con: int,
    ef_max: int,
    omp_threads: int,
    extra_args: list[str],
) -> tuple[float | None, str, int]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = [
        binary,
        "-build_only",
        "-dataset",
        dataset.dataset_arg,
        "-N",
        str(n_points),
        "-dataset_path",
        dataset.base_path,
        "-index_k",
        str(index_k),
        "-ef_con",
        str(ef_con),
        "-ef_max",
        str(ef_max),
        *extra_args,
    ]

    print(f"Running: {' '.join(cmd)}")
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )

    out = p.stdout or ""
    return parse_build_time_s(out), out, p.returncode


def build_only_cmd(
    *,
    binary: str,
    dataset: Dataset,
    n_points: int,
    index_k: int,
    ef_con: int,
    ef_max: int,
    extra_args: list[str],
) -> list[str]:
    return [
        binary,
        "-build_only",
        "-dataset",
        dataset.dataset_arg,
        "-N",
        str(n_points),
        "-dataset_path",
        dataset.base_path,
        "-index_k",
        str(index_k),
        "-ef_con",
        str(ef_con),
        "-ef_max",
        str(ef_max),
        *extra_args,
    ]


def main() -> int:
    serf_bin_default, hnsw_bin_default = default_binary_paths()

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000, help="Number of base vectors (N). Default: 1,000,000")
    ap.add_argument("--m", type=int, default=32, help="index_k (M). Default: 32")
    ap.add_argument("--k", type=int, default=400, help="ef_construction (K). Default: 400")
    ap.add_argument("--ef-max", type=int, default=500, help="ef_max. Default: 500")
    ap.add_argument("--threads", type=int, default=30, help="OMP_NUM_THREADS. Default: 30")
    ap.add_argument("--serf-bin", default=serf_bin_default, help="Path to serf_multithread binary")
    ap.add_argument("--hnsw-bin", default=hnsw_bin_default, help="Path to hnsw_multithread binary")
    ap.add_argument(
        "--out-dir",
        default=str(repo_root() / "results" / "exp4_ingestion"),
        help="Output directory for CSV/logs. Default: repo_root/results",
    )
    ap.add_argument(
        "--csv-name",
        default="build_time_comparison.csv",
        help="CSV filename under out-dir. Default: build_time_comparison.csv",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute")
    args, unknown = ap.parse_known_args()

    if unknown:
        print(f"Warning: ignoring unknown args: {unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / args.csv_name
    log_path = out_dir / f"build_time_buildonly_{ts}.log"

    if not args.dry_run:
        for b in (args.serf_bin, args.hnsw_bin):
            if not Path(b).exists():
                print(f"ERROR: binary not found: {b}")
                return 2

    rows: list[dict[str, object]] = []

    header = (
        f"Build-only benchmark @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"N={args.n}, M(index_k)={args.m}, K(ef_con)={args.k}, ef_max={args.ef_max}, threads={args.threads}\n"
        f"SeRF bin: {args.serf_bin}\n"
        f"HNSW bin: {args.hnsw_bin}\n"
    )
    print(header)

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(header)
        for ds in DATASETS:
            for method_name, bin_path in (
                ("serf", args.serf_bin),
                ("hnsw", args.hnsw_bin),
            ):
                if args.dry_run:
                    cmd = build_only_cmd(
                        binary=bin_path,
                        dataset=ds,
                        n_points=args.n,
                        index_k=args.m,
                        ef_con=args.k,
                        ef_max=args.ef_max,
                        extra_args=[],
                    )
                    print(f"OMP_NUM_THREADS={args.threads} " + " ".join(cmd))
                    continue

                build_time_s, output, rc = run_build_only(
                    binary=bin_path,
                    dataset=ds,
                    n_points=args.n,
                    index_k=args.m,
                    ef_con=args.k,
                    ef_max=args.ef_max,
                    omp_threads=args.threads,
                    extra_args=[],
                )

                lf.write("\n" + "=" * 80 + "\n")
                lf.write(f"[{method_name}] {ds.name} ({ds.dim}D)\n")
                lf.write(output)
                lf.write("\n")

                ips = (args.n / build_time_s) if (build_time_s and build_time_s > 0) else None
                rows.append(
                    {
                        "method": method_name,
                        "M": args.m,
                        "K": args.k,
                        "K_Search": "",
                        "range_pct": "",
                        "recall": "",
                        "qps": "",
                        "comps": "",
                        "dataset": f"{ds.name}-{ds.dim}",
                        "build_time": build_time_s,
                        "ips": ips,
                    }
                )

                bt_str = f"{build_time_s:.3f}s" if isinstance(build_time_s, float) else "N/A"
                print(f"[{method_name:<4}] {ds.name:<4} ({ds.dim:>4}D)  Build Index Time: {bt_str}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "dataset",
                "M",
                "K",
                "K_Search",
                "range_pct",
                "recall",
                "qps",
                "comps",
                "build_time",
                "ips",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV: {csv_path}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
