# D2 — Frozen-model task-completion, violation, instruction-stage, stop/resume baseline

See D2_summary.json, D2_baseline_verdict.json for the canonical verdict.

Key findings:
1. D1.8.2 is the canonical input. No supplementary rerun needed for partial D2.
2. 260/260 decisions, 257 non-zero, 3 all-zero (caused by speed→0 at full stop).
3. 13/13 handoffs in range, 13/13 valid startups.
4. D0 strict episode success: 0/13 (0%) - target 90% NOT met.
5. Primary failure: model cannot restart from speed=0 (confirmed by D1.8.3).
6. 5/5 D0 violation categories require supplementary rerun with extended instrumentation.
7. Single-GPU latency >150ms (D1.8.2: max 1670ms) - NOT a D2 regression.

D2 evaluation complete: true
Behavioral acceptance pass: false
Latency acceptance pass: false
D3 can proceed: true (with supplementary rerun)
D5/D6 can proceed: true (in parallel)
