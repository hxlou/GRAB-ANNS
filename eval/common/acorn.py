#!/usr/bin/env python3
"""Run the unmodified ACORN reference benchmark with paper parameters."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACORN_ROOT = ROOT / "third_party" / "ACORN"


def default_binary() -> Path:
    candidates = [
        ACORN_ROOT / "build" / "demos" / "test_acorn",
        ACORN_ROOT / "build-noinstall" / "demos" / "test_acorn",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=default_binary())
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=int, default=100)
    parser.add_argument("--dataset", default="paper")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--m-beta", type=int)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    m_beta = args.m_beta if args.m_beta is not None else 2 * args.m
    command = [
        str(args.binary),
        str(args.n),
        str(args.gamma),
        args.dataset,
        str(args.m),
        str(m_beta),
    ]
    print(" ".join(command))
    if args.dry_run:
        return 0
    if not args.binary.exists():
        raise FileNotFoundError(
            f"ACORN benchmark not found: {args.binary}. Build third_party/ACORN first."
        )

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    result = subprocess.run(
        command,
        cwd=ACORN_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(result.stdout, encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
