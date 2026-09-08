#!/usr/bin/env python3
"""Run a fresh, isolated CagraIndexOpt grid with a frozen verified library.

Existing output directories are refused. Source/index/data files are read-only.
The frozen library is preloaded because the benchmark has a legacy DT_RPATH.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd

ITOPKS = (32, 64, 96, 128, 192, 256, 384, 512)
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16)
ITERS = (3, 5, 8, 10, 12, 15, 20, 30, 50, 80, 100, 150, 200)
DATASETS = {
    "gist": ("GIST-960", "data/GIST1M/gist_base.fvecs"),
    "deep": ("DEEP-96", "data/deep_base.fvecs"),
    "sift": ("SIFT-128", "data/sift.fvecs"),
    "wit": ("WIT-2048", "data/wit-image.fvecs"),
}
RATIOS = (0.01, 0.10, 0.20, 1.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configs(degree):
    # Local queries clamp itopk to 128. Range/global clamp to 64.
    # Restrict all modes to the smallest common implemented queue limit.
    return [(k, w, i) for k, w, i in itertools.product(ITOPKS, WIDTHS, ITERS)
            if max(128, k) + w * degree <= 1024]


def normalize(frame, label, degree, bitlen, source, phase, rounds):
    if frame.empty or frame[["Recall", "QPS"]].isna().any().any():
        raise RuntimeError(f"Invalid results: {source}")
    if not ((frame.QPS > 0) & frame.Recall.between(0, 100)).all():
        raise RuntimeError(f"Out-of-range metrics: {source}")
    return pd.DataFrame({
        "dataset": DATASETS[label][0], "method": "lightCagra",
        "range_pct": frame.Ratio * 100, "recall": frame.Recall / 100,
        "qps": frame.QPS, "M": degree, "K": 10, "K_Search": frame.Itopk,
        "search_width": frame.Width, "max_iterations": frame.Iter,
        "local_degree": frame.LocalDegree, "mode": frame.Mode,
        "batch": frame.Batch, "avg_ms": frame.AvgMs,
        "build_time": frame.BuildMs, "hash_bitlen": bitlen,
        "rounds": rounds, "phase": phase, "source_csv": str(source),
        "algorithm_commit": "c804784fb5e8f853360bd0e2672991476f5efbd5",
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/lhx/lightCagra"))
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root, output, snapshot = (x.resolve() for x in (args.root, args.output, args.snapshot))
    jobs = [(label, degree, bitlen) for label in DATASETS
            for degree in (32, 64) for bitlen in (12, 13, 14)]
    expected = sum(4 * len(configs(degree)) for _, degree, _ in jobs)
    print(json.dumps({"config_count_by_degree": {d: len(configs(d)) for d in (32, 64)},
                      "jobs": len(jobs), "expected_rows": expected}), flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    for label, degree, _ in jobs:
        for path in (root / DATASETS[label][1],
                     root / f"build/{label}_1m_cagra_opt_deg{degree}_inter512_m64_nprobe2048.idx"):
            if not path.is_file():
                raise FileNotFoundError(path)
    committed = subprocess.check_output(["git", "show", "c804784:src/index/search.cu"], cwd=root)
    if hashlib.sha256(committed).hexdigest() != digest(snapshot / "search.cu"):
        raise RuntimeError("Frozen source does not match the stable algorithm commit")
    output.mkdir(parents=True)
    shutil.copytree(snapshot, output / "frozen")
    shutil.copy2(__file__, output / Path(__file__).name)
    (output / "runs").mkdir()
    (output / "smoke").mkdir()
    env = os.environ.copy()
    # Record and clear tuning overrides so every result uses committed defaults.
    removed = {k: env.pop(k) for k in list(env) if k.startswith("CAGRA_")}
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LD_PRELOAD"] = str(output / "frozen/liblight_cagra.so")
    executable = output / "frozen/test_cagra_opt_small_batch_auto"
    linkage = subprocess.check_output(["ldd", str(executable)], env=env, text=True)
    if env["LD_PRELOAD"] not in linkage or "build_gpu0/liblight_cagra" in linkage:
        raise RuntimeError(f"Wrong library selected: {linkage}")
    (output / "linkage.txt").write_text(linkage)
    manifest = {
        "algorithm_commit": "c804784fb5e8f853360bd0e2672991476f5efbd5",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "itopks": ITOPKS, "widths": WIDTHS, "iterations": ITERS,
        "hash_bitlens": [12, 13, 14], "expected_rows": expected,
        "rounds": args.rounds, "batch": 1000, "n": 1000000, "buckets": 100,
        "ratios": RATIOS, "random_range_per_round": False,
        "query_protocol": "Existing benchmark deterministic range/query sampling from base; search seeds random per call",
        "cleared_tuning_overrides": removed,
        "snapshot_sha256": {f.name: digest(f) for f in (output / "frozen").iterdir() if f.is_file()},
        "jobs": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []

    def run_job(label, degree, bitlen, grid, phase, rounds):
        tag = f"{label}_d{degree}_h{bitlen}"
        csv = output / phase / f"{tag}.csv"
        log = output / phase / f"{tag}.log"
        command = [str(executable), "--data", str(root / DATASETS[label][1]),
                   "--n", "1000000", "--buckets", "100", "--degree", str(degree),
                   "--local-degree", str(degree // 2), "--intermediate-degree", "512",
                   "--remote-pq-m", "64", "--remote-nprobe", "2048",
                   "--rounds", str(rounds), "--batches", "1000",
                   "--ratios", ",".join(map(str, RATIOS)),
                   "--search-configs", ",".join(":".join(map(str, c)) for c in grid),
                   "--algo", "baseline", "--gt-device", "gpu", "--hash-bitlen", str(bitlen),
                   "--load-index", str(root / f"build/{label}_1m_cagra_opt_deg{degree}_inter512_m64_nprobe2048.idx"),
                   "--csv", str(csv)]
        start = time.monotonic()
        print(f"{time.strftime('%F %T')} start {phase}/{tag}: {len(grid)*4} rows", flush=True)
        with log.open("x") as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        contents = log.read_text()
        if any(s in contents for s in ("ERROR:", "CUDA Error", "illegal memory", "Unsupported queue")):
            raise RuntimeError(f"Kernel error reported in {log}")
        frame = pd.read_csv(csv)
        keys = ["Itopk", "Width", "Iter", "Ratio", "Batch"]
        if len(frame) != 4 * len(grid) or frame.duplicated(keys).any():
            raise RuntimeError(f"Unexpected row count/duplicate config in {csv}: {len(frame)}")
        print(f"{time.strftime('%F %T')} complete {phase}/{tag}: {len(frame)} rows, {time.monotonic()-start:.1f}s", flush=True)
        return normalize(frame, label, degree, bitlen, csv, phase, rounds)

    # Validate largest supported queue and newly added shared hash on all dimensions.
    smoke_grid = [(64, 1, 10), (128, 12, 20), (256, 12, 50), (512, 8, 200)]
    for label in DATASETS:
        run_job(label, 64, 13, smoke_grid, "smoke", 2)
    for label, degree, bitlen in jobs:
        result = run_job(label, degree, bitlen, configs(degree), "runs", args.rounds)
        rows.append(result)
        manifest["jobs"].append({"label": label, "degree": degree, "hash_bitlen": bitlen,
                                 "rows": len(result), "finished": time.strftime("%F %T")})
        (output / "progress.json").write_text(json.dumps(manifest, indent=2) + "\n")
    final = pd.concat(rows, ignore_index=True)
    if len(final) != expected:
        raise RuntimeError(f"Incomplete final: {len(final)} vs {expected}")
    final.to_csv(output / "parato_cagra.csv", index=False, mode="x")
    final.to_csv(output / "pareto_cagra.csv", index=False, mode="x")
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "completed.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMPLETE: {len(final)} fresh rows in {output}", flush=True)


if __name__ == "__main__":
    main()
