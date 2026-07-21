# D2.1 Evidence Packages

For every D2.1 episode, an evidence package is produced at
`output/carla_acceptance/D2_1_fully_instrumented_baseline/evidence_packages/<episode_id>/package.json`.

The package contains:

- `episode_summary` — frame count, decision count, control-source counts,
  min/max/mean speed, max route progress, emitted stages, violation counts
- `instruction_stage_timeline` — first-frame of each new stage
- `violation_events` — per-category event lists with frame + magnitude
- `stop_resume_timeline` — stops and resumes with frame + speed
- `route_progress_timeline` — frame + progress snapshots
- `speed_timeline` — frame + speed_mps
- `first_failure_evidence` — first violation detected per category
- `terminal_state` — task terminal reason
- `control_source_timeline` — frame + control_source

Keyframes (PNG references) are produced by the gateway (`per_decision_images/`)
and are linked back from the relevant timeline entries.  Video production is
NOT in D2.1 scope (D4 produces D4-quality video).
