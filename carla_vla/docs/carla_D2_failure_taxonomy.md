# D2 failure taxonomy

Primary cause: MODEL_REMAINED_EXACT_ALL_ZERO
- 3/260 all-zero outputs in D1.8.2
- Root cause: speed=0 → all-zero (D1.7 speed-gating)
- D1.8.3 confirmed: 0/1 successful restart from full stop

Secondary: D0 violations and latency cannot be evaluated
without supplementary rerun (no collision sensor, no light state, no
marking state, no lane membership, no terminal state snapshot).
