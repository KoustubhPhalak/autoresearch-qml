# NTangled Parameter Classification Research
Goal: Outperform the baseline by optimizing the Angle Embedding pattern.

## The Challenge
The dataset contains classical weight vectors from the Schatzki NTangled repository. 
We need to determine if a set of weights produces Low or High entanglement.

## Research Directions for Claude
1. **Embedding Iterations:** Instead of a single RX-RY layer, try 'Data Re-uploading'. Apply RX/RY, then a small variational block, then repeat the RX/RY embedding.
2. **Rotation Diversity:** Experiment with different gate sequences in the embedding (e.g., RX-RZ vs RX-RY-RZ).
3. **Entanglement Topology:** In the variational layers, move beyond the 'Ring CNOT' to a 'All-to-All' or 'QCNN' pooling structure.
4. **Encoding Ratios:** Test if using 1 feature per qubit (RX) is better than packing 3 features per qubit (RX-RY-RZ).

## Constraints
- Max training time per run: 300s.
- Target: Improve TEST_ACC from 0.55 to 0.85+.
- Do NOT touch prepare.py.