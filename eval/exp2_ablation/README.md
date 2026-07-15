# Exp2: GRAB-ANNS Ablation

The ablation sweeps graph degree, candidate queue size, and maximum search
iterations at range selectivities 1%, 10%, 20%, and 100%.

```bash
mkdir -p results/exp2_ablation
./build/grab_exp2_ablation "$GRAB_DEEP_BASE" results/exp2_ablation/grab.csv
```
