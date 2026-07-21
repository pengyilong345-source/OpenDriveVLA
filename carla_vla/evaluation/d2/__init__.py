"""D2 evaluators for frozen-model CARLA task-completion baseline.

Operating principle:
- Reads existing D1.8.2 per-frame logs.
- Computes metrics that are derivable from stored data.
- Explicitly marks metrics that require supplementary reruns.

D2 does NOT modify the checkpoint, weights, prompt, controller, or safety policy.
"""
