# BoseHubbard 3-Phase Classification Notes

## Dataset
- BoseHubbard 1x4 open+closed combined: 200 samples (139 train, 61 test)
- 3 classes: superfluid-like (U<2.5), critical-like (2.5≤U≤4.5), mott-like (U>4.5)
- 8-qubit Hilbert space (256-dim statevectors)
- Class balance: 72/56/72

## Results

| Iter | Description | TEST_ACC | Notes |
|------|-------------|----------|-------|
| 0 | Baseline: Z-probs → Linear(256,4) → 2-layer RY-RZ ring CNOT → Linear(4,3) | 0.6393 | Converges quickly, stuck at ~64% |
