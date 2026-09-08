# Expanded CagraIndexOpt sweep (2026-09-08)

The complete exploratory sweep contains **73,632 freshly measured points** on
DEEP-96, SIFT-128, GIST-960, and WIT-2048. Every dataset uses 1M vectors,
100 contiguous labels of 10K vectors, degree 32/64 cached indexes, and
1%, 10%, 20%, and 100% ranges. All 24 jobs completed and passed row-count,
duplicate-configuration, metric-range, and runtime-error checks.

The algorithm is frozen at `c804784fb5e8f853360bd0e2672991476f5efbd5`.
The runner copies the existing verified binary/library/source snapshot into
the new experiment directory, checks source provenance and resolved linkage,
and uses `LD_PRELOAD` because the benchmark has a legacy `DT_RPATH`.
It does not rebuild or mutate the cached indexes, data files, or algorithm
working tree. The uncommitted guarded 50% hash-reset experiment is excluded.

## Parameter grid

- Candidate queue (`itopk`): 32, 64, 96, 128, 192, 256, 384, 512.
- Expansion width: 1, 2, 3, 4, 6, 8, 12, 16.
- Maximum iterations: 3, 5, 8, 10, 12, 15, 20, 30, 50, 80, 100, 150, 200.
- Hash bit length: 12, 13, 14.
- Batch size: 1,000; five timed rounds in exploration.
- Capacity guard: `max(128, itopk) + width * degree <= 1024`.

The guard accounts for the local kernel's minimum effective queue of 128 and
the common implemented 1024-slot sort limit. Degree 32 admits 832 search
configurations; degree 64 admits 702. The 32/64/96/128 requested `itopk`
values are retained because they also affect seed selection, even when the
effective search queue is clamped internally.

Before the grid, a 64-point smoke check exercised all four dimensions with
new hash13 and the largest supported queue. The full grid completed from
10:14:42 to 11:50:03 server time; smoke checks started at 10:10:59.

## Reproduction and outputs

Run from the project root with the `faiss` Conda environment. Always choose
a new output path; the runner refuses an existing directory.

```sh
python scripts/run_expanded_final_sweep.py \
  --snapshot experiments/global_hash_reset50_guarded_after_c804784_v1/baseline \
  --output experiments/expanded_c804784_final_sweep_v1
```

The completed output is
`experiments/expanded_c804784_final_sweep_v1/parato_cagra.csv`.
`runs/` retains every raw CSV and log; `frozen/`, `manifest.json`,
`linkage.txt`, and `completed.json` record provenance and completion.
No prior experimental output or raw data was deleted or overwritten.
With `--load-index`, the benchmark's raw `BuildMs` is index **load time**,
not construction time; do not use it to report build performance.

## Interpretation and follow-up

Repeat verification of Pareto candidates and previous 90%/95% winners is
queued separately: three passes with independently drawn search seeds,
20 timed rounds per pass, and median QPS/recall aggregation. The exploratory
maximum should not be presented as the verified final result.

The historical official cuVS baseline is preserved, but its measurement
protocol differs: centered windows and different query sampling, with
device-resident search timing. Our benchmark measures the public API,
including host preparation and transfers. A sampled audit of actual saved
indexes found that SIFT matched both input files at all 1,007 sampled row
positions, whereas WIT matched the original file at all 1,007 positions
and the cuVS input at none. Cross-method plots are reference comparisons;
WIT specifically has an incompatible row mapping/split for the label scheme.

Source review additionally identified candidate-ID/batch-offset handling and
sort-layout problems in remote-edge exact reranking. A small isolated CUDA
diagnostic is queued after performance measurement, without rebuilding or
modifying any saved index. Final evidence and prioritized optimization
recommendations will be recorded with the repeat-verified results.
