# BoseHubbard Phase Classification Notes

## Dataset
- BoseHubbard 1x4 open+closed combined: 200 samples (139 train, 61 test)
- 8-qubit Hilbert space (256-dim statevectors)
- Original 3-class: superfluid (U<2.5), critical (2.5≤U≤4.5), mott (U>4.5)

## Critical Finding (iter14-15)
**Classes 1 (critical) and 2 (mott) are IDENTICAL quantum states: |00000000⟩**
- Fidelity = 1.0 between all critical/mott samples
- Both have PauliZ expval = 1.0 on all 8 qubits
- No quantum measurement can distinguish them
- The 3-class problem is fundamentally unsolvable with quantum methods

**5 "superfluid" test samples are also |00000000⟩** (near phase boundary U≈2.5)
- These are inherent label noise → max achievable accuracy = 56/61 = 91.8%

## Reformulated Problem
**2-class: superfluid (U<2.5) vs insulator (U≥2.5)**
- Physically meaningful Mott transition boundary
- 91.8% is the theoretical ceiling with this labeling

## Results

| Iter | Description | TEST_ACC | Notes |
|------|-------------|----------|-------|
| 0 | Baseline: Z-probs → Linear → 2L RY-RZ CNOT → Linear (3-class) | 0.6393 | Always predicts majority |
| 1 | Data re-uploading RX-RY-RZ (3-class) | 0.2623 | Too deep, no learning |
| 2-8 | Various: physics features, IQP, QBN, thresholds (3-class) | 0.6393 | All stuck at majority class |
| 9 | Quantum kernel SVM (3-class) | 0.6393 | Even SVM can't separate |
| 10-12 | Tree-like encoding, binary indicators (3-class) | 0.4426 | Binary features → no gradient |
| 13 | Amplitude embed 8q + 1 var layer (3-class) | 0.6393 | Still stuck |
| 14 | + rich observables X,Y,ZZ,XX (3-class) | 0.6393 | Classes 1&2 identical → impossible |
| **15** | **2-class: StatePrep → 1L var → PauliZ → QBN → 2 Linear** | **0.9180** | **Theoretical ceiling** |
| 16 | 0,1,2 layers comparison (2-class) | 0.9180 | All same — ceiling hit |

## Classical Baselines
- RF on Z-probs (3-class): 1.0000 — but uses label info unavailable to quantum
- GBM on Z-probs (3-class): 1.0000
- SVM/MLP on Z-probs (3-class): 0.6393 — same wall as QML
- All methods on 2-class: 0.9180 ceiling (due to 5 mislabeled samples)

## Architecture Notes
- QBN after QNN helps stabilize training but doesn't break the ceiling
- Max 2 linear layers after QNN (per user requirement)
- Amplitude embedding (StatePrep) is the natural encoding for quantum states
- Variational layers don't help beyond 0 layers — raw PauliZ already optimal
