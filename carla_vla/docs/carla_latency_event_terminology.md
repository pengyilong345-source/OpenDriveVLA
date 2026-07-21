# Latency event terminology (D0.1.1)

Four separate event fields replace the old merged "timeout/hang":

| field | definition |
|---|---|
| `deadline_miss` | D0 T0-T10 total latency > 150 ms |
| `gateway_response_timeout` | request exceeds the configured gateway waiting threshold |
| `server_hang` | server heartbeat stops or request never completes |
| `process_restart` | process boot_id changes |

Legacy logs that merged these are marked `unknown_from_legacy_log`.
