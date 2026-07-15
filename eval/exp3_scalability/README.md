# Exp3: Scalability

This experiment scales DEEP-96 from 1M to 10M vectors. The main comparison
uses 10% selectivity and fixed parameters for every method.

```bash
mkdir -p results/exp3_scalability
./build/grab_exp3_scalability \
  "$GRAB_DEEP10M_BASE" results/exp3_scalability/grab.csv
python3 eval/exp3_scalability/serf_hnsw.py
python3 eval/exp3_scalability/acorn.py --gamma 100 --m 32
python3 eval/exp3_scalability/milvus.py
```
