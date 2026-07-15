# Evaluation

This directory contains the experiment drivers used for the paper. The layout
is experiment-first so every method evaluated in a figure is visible in one
place.

```text
eval/
  exp1_search_efficiency/  # Recall--QPS Pareto frontiers
  exp2_ablation/           # GRAB-ANNS parameter ablations
  exp3_scalability/        # DEEP-96, 1M--10M
  exp4_ingestion/          # 1M-vector ingestion time
  common/                  # shared adapters
```

Exp1, Exp3, and Exp4 include GRAB-ANNS, ACORN, SeRF, HNSW-Filter, and
Milvus. Exp2 is an ablation of GRAB-ANNS itself.

## Setup

Clone third-party sources and build GRAB-ANNS and SeRF:

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake -S third_party/SeRF -B third_party/SeRF/build -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/SeRF/build -j --target \
  serf_multithread hnsw_multithread

cmake -S third_party/ACORN -B third_party/ACORN/build \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_SKIP_INSTALL_RULES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/ACORN/build -j --target test_acorn
```

Configure datasets without editing scripts:

```bash
cp eval/datasets.env.example eval/datasets.env
# Edit eval/datasets.env, then:
source eval/datasets.env
```

Milvus drivers require `pymilvus` and a running Milvus standalone service.
ACORN uses the unmodified reference implementation under `third_party/ACORN`;
its `paper` dataset files must be prepared according to the upstream demo.

All generated indexes, logs, and CSV files belong under `results/` and are not
source-controlled.
