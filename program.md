# NTangled Parameter Classification Research
Goal: Outperform the baseline by optimizing the Angle Embedding pattern.

## The Challenge
The dataset contains classical weight vectors from the Schatzki NTangled repository. 
We need to determine if a set of weights produces Low or High entanglement.

## Research Directions for Claude
1. **Embedding Iterations:** Instead of a single RX-RY layer, try 'Data Re-uploading'. Apply RX/RY, then a small variational block, then repeat the RX/RY embedding. Don't necessarily restrict yourself to these, expand and be creative as you see fit.
2. **Rotation Diversity:** Experiment with different gate sequences in the embedding (e.g., RX-RZ vs RX-RY-RZ).
3. **Entanglement Topology:** In the variational layers, move beyond the 'Ring CNOT' to a 'All-to-All' or 'QCNN' pooling structure. If these don't work, go beyond these and try.
4. **Encoding Ratios:** Test if using 1 feature per qubit (RX) is better than packing 3 features per qubit (RX-RY-RZ). Again, be flexible beyond this as well.
5. **Updating stats and notes** Continuously update NOTES.md and progress.png as per your new/novel findings.

## Constraints
- Max training time per run: 300s.
- Target: Improve TEST_ACC from 0.55 to 0.85+.
- Do NOT touch prepare.py.
- NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?"or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
- Don't get too stuck up on an idea. If an idea you are trying doesn't work for maximum 4 iterations, drop that idea and think of something else, unless I manually tell you otherwise.