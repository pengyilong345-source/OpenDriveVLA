# D0.1.1 — Stop/resume protocol hardening

Pure additive/clarifying amendment to v1.1.0. Key additions:

1. **Strict handoff-speed boundary**: 5.0 ≤ speed ≤ 8.0, tolerance = 0.
2. **Terminology separation**: deadline_miss / gateway_response_timeout /
   server_hang / process_restart are 4 separate fields.
3. **Stop/resume definitions**: full_stop (speed ≤ 0.10 for ≥ 1.0 s),
   resume_success (non-zero trajectory + speed > 1.0 + advances > 2.0 m),
   resume_timeout = 5.0 s.
4. **Legitimate-stop vs abnormal-all-zero**: scenario-grounded hazard state,
   not based on whether safety braking fired.

Original D0 success formula, violations, semantic alignment, latency, and
warmup exclusions are all unchanged.
