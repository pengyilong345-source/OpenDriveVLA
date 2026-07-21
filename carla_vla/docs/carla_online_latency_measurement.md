# Online closed-loop latency measurement

## 1. Timestamps

All timestamps use **the same-host monotonic clock** via `time.monotonic_ns()`.
Both processes run on the same machine, so values are comparable across PIDs
(unlike `time.time()` under NTP adjustment).

| stage | name | where it runs | what happens |
|---|---|---|---|
| T0 | sensor-frame-ready | gateway | six CARLA sensor images are ready and asserted to share `image.frame` |
| T1 | shm-publish-done | gateway | camera bytes written + header bumped; Unix-socket REQUEST envelope sent |
| T2 | inference-receive | server | server received the REQUEST envelope AND successfully validated the shm frame (sha256 + same-frame) |
| T3 | preprocess-done | server | image preprocessing done (RGB → BGR-mean) and CUDA tensor allocated |
| T4 | vision-done | server | UniAD / vision tower forward complete (backbone + BEV) |
| T5 | prompt-done | server | official-compatible prompt built and tokenized |
| T6 | generate-done | server | language model auto-regressive decode complete |
| T7 | parse-done | server | trajectory parsed and validity flags set (parse_failure / all_zero) |
| T8 | controller-done | server | pure-pursuit + speed PI has emitted (steer, throttle, brake) |
| T9 | gateway-receive | gateway | RESPONSE envelope received over the Unix socket |
| T10 | control-applied | gateway | `ego.apply_control()` returned |

The primary total:
```
total_decision_latency_ms = (T10 - T0) / 1e6
```

## 2. Per-module deltas

```
sensor_publish_ms       = T1 - T0
IPC_to_inference_ms     = T2 - T1
preprocess_transfer_ms  = T3 - T2
vision_ms               = T4 - T3
prompt_tokenization_ms  = T5 - T4
generation_ms           = T6 - T5
parse_validation_ms     = T7 - T6
controller_ms           = T8 - T7
IPC_to_carla_ms         = T9 - T8
apply_control_ms        = T10 - T9
```

These are reported as the `per_module_ms` block in `latency_breakdown.json`,
each with count / mean / median / p90 / p95 / p99 / max / min / std.

## 3. Deadline + strict verdict

The hard deadline is **150 ms** end-to-end (T10 - T0). The strict verdict is
defined as:

```
strict_verdict_pass  <=>  every recorded cycle has total_decision_latency_ms <= 150
```

The summary reports the strict verdict as a Boolean AND additionally:

- **mean, median, p90, p95, p99, max** (in ms),
- **deadline_miss_count** and **deadline_miss_rate** (count / valid_total),
- **per-frame deadline_miss list** in `deadline_misses.json`.

A single miss fails the strict verdict. Percentile-only verdicts are
explicitly disallowed by the spec and are NOT used as the primary signal.

## 4. Warm-up separation

- The server's **first three inferences are dummy warm-up** runs (zero
  tensors). They populate GPU caches and DeepSpeed bookkeeping.
- Warm-up cycles are recorded in `health_server.jsonl` as
  `"warmup_done": true` and are NOT counted in `latency_breakdown.json`
  totals (`n_records` excludes warm-up).
- Cold-start is reported as the first cycle's `total_decision_latency_ms`
  and is reported separately as `first_cycle_ms` in
  `latency_breakdown.json`.

## 5. Asynchronous GPU handling

- `torch.cuda.synchronize()` is called BEFORE reading T3, T6, and T8 in
  the server. Without this, an asynchronous op enqueued on the GPU would
  let the Python interpreter record a smaller-than-realistic latency.
- No profiling is done for an offline replay path. All numbers come from
  the actual live inference.

## 6. Where each timestamp is recorded

| stage | recorded by | encoded as |
|---|---|---|
| T0 | `carla_gateway_py37.run_episode()` (immediately after `read_same_frame` and AFTER `world.tick()`) | absolute ns in `stages_ns.T0` |
| T1 | same, after `fw.publish()` and `send_envelope()` | absolute ns in `stages_ns.T1` |
| T2-T8 | `opendrivevla_server.handle_request()` (after `fr.read_latest()`, before `_move_ud`) | returned in `latencies_ms` (`T2_ns`..`T8_ns`) over the socket |
| T9 | gateway, after `recv_envelope()` returns non-None | absolute ns in `stages_ns.T9` |
| T10 | gateway, after `ego.apply_control()` returns | absolute ns in `stages_ns.T10` |

All stages are in the same monotonic clock domain.

## 7. What is NOT in the latency budget

The following are deliberately excluded from the decision-cycle latency:

- Disk I/O (none — `/dev/shm` is RAM-backed).
- Heartbeat writes (background, every 1 s, not on the critical path).
- Control visualization (none — videos / plots are an offline postprocess).
- Process startup (handled separately as cold-start).
- CARLA world tick physics steps. The replanning step is invoked after
  the tick. The tick duration is reported by CARLA itself via the
  `sim_t` delta, but is NOT in the decision-cycle latency (the spec
  measures "synchronized sensor batch ready -> corresponding control
  command applied", which is from T0 to T10).

## 8. Headline summary fields in `latency_breakdown.json`

```jsonc
{
  "deadline_ms": 150.0,
  "totals": {
    "count": 245,
    "mean":     1320.5,  // ms
    "median":   1280.4,
    "p90":      1450.2,
    "p95":      1480.7,
    "p99":      1495.0,
    "max":      1605.0,
    "min":      1100.3,
    "std":       88.4
  },
  "n_records":  260,
  "n_valid":    250,
  "n_stale":     5,
  "n_dropped":   5,
  "deadline_miss_count":  260,
  "deadline_miss_rate":    1.00,
  "strict_verdict_pass":   false,
  "first_cycle_ms":        1850.2,   // cold-start (NOT in mean)
  "per_module_ms": {
    "sensor_publish_ms":      {"mean": 235.6, ...},
    "IPC_to_inference_ms":    {"mean": 0.4, ...},
    "preprocess_transfer_ms": {"mean": 250.0, ...},
    "vision_ms":              {"mean": 0.2, ...},
    "prompt_tokenization_ms": {"mean": 2.1, ...},
    "generation_ms":          {"mean": 1097.9, ...},
    "parse_validation_ms":    {"mean": 0.4, ...},
    "controller_ms":          {"mean": 0.05, ...}
  }
}
```

The `generation_ms` is by far the dominant cost (~ 1.1 s, vs all other
modules combined < 0.3 s). The 150 ms deadline is therefore **never
met at this checkpoint size** on this host. The strict verdict is used
verbatim and the report shows the over-budget breakdown honestly.
