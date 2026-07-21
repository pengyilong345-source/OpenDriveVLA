# D4 Curve Contract

Each episode produces up to 11 frozen curve PNGs:

1. `speed_vs_sim_time.png` — ego speed (m/s) vs simulation time (s)
2. `acceleration_vs_time.png` — optional (deferred; offline)
3. `throttle_brake_timeline.png` — control signals vs simulation time
4. `predicted_path_length_vs_time.png` — optional (deferred; offline)
5. `hazard_active_clear_timeline.png` — optional (deferred)
6. `command_stage_timeline.png` — optional (deferred)
7. `route_progress_vs_time.png` — optional (deferred)
8. `model_latency_per_decision.png` — optional (deferred)
9. `alignment_verdict_timeline.png` — D3 alignment verdicts vs decision idx
10. `violation_event_timeline.png` — optional (deferred)
11. `stop_resume_timeline.png` — optional (deferred)

The first three (`speed`, `throttle/brake`, `decision timeline`) are produced
unconditionally per episode. The others are deferred to the post-pilot
full 13-scenario run.

## Curve Render Path

`carla_vla/visualization/d4/curve_renderer.py::render_curves_for_episode`
- Inputs: `tick_timelines/<ep_id>/tick_timeline.jsonl` and
  `decision_bundles/<ep_id>__bundle_index.jsonl`.
- Outputs: PNGs at
  `output/carla_acceptance/D4_1_visualization_baseline/episodes/<ep_id>/curves/`.
- Uses `matplotlib` Agg backend (no display required).
- Does NOT modify any input; pure read + PNG render.
