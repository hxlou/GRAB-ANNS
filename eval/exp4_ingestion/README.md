# Exp4: Index Ingestion

The ingestion benchmark measures the time required to process 1M vectors on
DEEP-96, SIFT-128, GIST-960, and WIT-2048.

```bash
mkdir -p results/exp4_ingestion
./build/grab_exp4_build "$GRAB_DEEP_BASE" DEEP-96
./build/grab_exp4_insert \
  "$GRAB_SIFT_BASE" results/exp4_ingestion/grab_insert.csv
python3 eval/exp4_ingestion/serf_hnsw.py
python3 eval/exp4_ingestion/acorn.py --gamma 100 --m 32
python3 eval/exp4_ingestion/milvus.py
```

Milvus runs against the service defined in `eval/common/milvus-compose.yml`.
