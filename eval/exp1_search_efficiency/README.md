# Exp1: Search Efficiency

This experiment constructs Recall@10--QPS Pareto frontiers on DEEP-96,
SIFT-128, GIST-960, and WIT-2048 at 1%, 10%, 20%, and 100% selectivity.

```bash
mkdir -p results/exp1_search_efficiency
./build/grab_exp1_search_efficiency "$GRAB_SIFT_BASE"
python3 eval/exp1_search_efficiency/serf_hnsw.py
python3 eval/exp1_search_efficiency/acorn.py --gamma 100 --m 32
python3 eval/exp1_search_efficiency/milvus.py
```

Run the GRAB executable once per dataset. ACORN's `paper` input must be
prepared in the reference repository format before invoking its adapter.
