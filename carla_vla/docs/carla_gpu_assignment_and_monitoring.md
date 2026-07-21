# GPU assignment and monitoring (D1.6)

## 1. Physical GPUs

This host has two identical NVIDIA GeForce RTX 4090 (24 GB each),
connected via PCIe and NUMA:

```
$ nvidia-smi topo -m
GPU0  X   SYS  SYS  SYS  0-31,64-95   0    N/A
GPU1  SYS  X   SYS  SYS  32-63,96-127 1    N/A
```

GPU 0 and GPU 1 live on different NUMA domains. Memory is not shared
between them.

## 2. How CARLA is bound to GPU 0

CARLA UE4 is launched by the `carlauser` system account with:

```
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000
```

The `my_nvidia_icd.json` Vulkan ICD file is bound to `libEGL_nvidia.so.0`,
which under no `VK_ICD_FILENAMES` from nvidia_select would default to GPU 0
in this host's wiring (X server + default seat 0). `nvidia-smi -q -d 0`
shows the CARLA compute process at ~8402 MiB on GPU 0.

`CUDA_VISIBLE_DEVICES` does **not** affect CARLA's Vulkan renderer, so
binding the inference server to GPU 1 via `CUDA_VISIBLE_DEVICES=1`
leaves CARLA's GPU 0 placement unchanged.

## 3. How the inference server is bound to GPU 1

The orchestrator passes `CUDA_VISIBLE_DEVICES=1` to the
`opendrivevla_server` subprocess via `subprocess.Popen(env=...)`. Inside
the server:

- `CUDA_VISIBLE_DEVICES=1` is in `os.environ`.
- The server's `_setup_for_replay()` logs:
  ```
  [odvla-server] CUDA_VISIBLE_DEVICES=1
  [odvla-server] local cuda:0 name=... uuid=GPU-dbe4d243-... total=24564MiB allocated=...
  ```
- `torch.cuda.current_device()` returns 0 (the only visible device).
- `torch.cuda.get_device_properties(0).uuid` returns the **physical GPU 1
  UUID**, confirming the remap.

The orchestrator CLI exposes `--inference-gpu-id 1` (default 1) and
`--carla-gpu-id 0` (default 0) for future tunability.

## 4. Why we did NOT add `CUDA_VISIBLE_DEVICES=0` to CARLA

CARLA's renderer uses Vulkan, not CUDA. Adding `CUDA_VISIBLE_DEVICES=0`
on the inference process hides GPU 1 from the inference server. It does
NOT affect CARLA's Vulkan picker because Vulkan uses the
`VK_ICD_FILENAMES` + `XDG_RUNTIME_DIR` + X seat, not CUDA. So we leave
the CARLA launch env untouched.

## 5. Verifying actual placement

`gpu_process_monitor.py` writes:
- `gpu_monitor.jsonl` — one row per cadence (200 ms default)
- `gpu_assignment_verification.json` — aggregate summary
- `process_gpu_map.json` — per-GPU process mapping
- `gpu_assignment_timeline.csv` — long-form time series

After the 3-scenario D1.6 smoke:

```
GPU 0 (a09d03e8-...): mem 8402 MiB, compute app pid 661698 (CarlaUE4)
GPU 1 (dbe4d243-...): mem 3473 MiB, compute app pid 896591 (Python inference)
```

Both placements confirmed via `nvidia-smi --query-compute-apps` while the
smoke was running.

## 6. Why we did NOT use a container

`carlauser` already runs CARLA in a specific environment with a
custom Vulkan ICD file. Containerizing CARLA just to bind it to GPU 0
would introduce a second system (nvidia-container-toolkit) and additional
overhead. The existing launch script is the verified baseline.

`CUDA_VISIBLE_DEVICES=1` on the inference server achieves the
isolation we need. If D2 requires stronger isolation (e.g., a faulty
NVIDIA driver hangs the renderer), we can revisit
`docker run --gpus '"device=0"'` for CARLA.

## 7. Monitoring tool

`carla_vla/online/gpu_process_monitor.py`:
- Polls `nvidia-smi --query-gpu=index,...` at 100-200 ms cadence
- Polls `nvidia-smi --query-compute-apps=...` at the same cadence
- Logs:
  - `per_gpu`: per-index memory, GPU util, memory util, power, temp
  - `per_process`: per-PID process name, GPU UUID, memory used
- Writes `gpu_monitor.jsonl` (one row per cycle) and
  `gpu_assignment_verification.json` (aggregate).

Usage:
```
python -m carla_vla.online.gpu_process_monitor \
  --duration 600 --cadence 0.5 \
  --output /path/to/gpu_monitor.jsonl \
  --summary /path/to/gpu_assignment_verification.json
```

## 8. Failure modes

| failure | detection | recovery |
|---|---|---|
| CARLA process on wrong GPU | `nvidia-smi --query-compute-apps` shows `[Not Found]:PID` on the unexpected UUID | stop the orchestrator; relaunch CARLA explicitly with the verified Vulkan ICD file |
| Inference process on wrong GPU | log line `[odvla-server] local cuda:0 ... uuid=...` does not match the expected GPU 1 UUID | check that `CUDA_VISIBLE_DEVICES=1` was passed to the subprocess via `server_env` in the orchestrator |
| Both processes on same GPU | UUIDs collide in `processes_seen_per_gpu_uuid` | inspect `VK_ICD_FILENAMES` for CARLA and `CUDA_VISIBLE_DEVICES` for the inference process |

## 9. Files

- `carla_vla/online/gpu_inventory.py` — static snapshot
- `carla_vla/online/gpu_process_monitor.py` — live monitor
- `output/carla_acceptance/D1_6_dual_gpu_validation/gpu_inventory.json`
- `output/carla_acceptance/D1_6_dual_gpu_validation/gpu_assignment_verification.json`
- `output/carla_acceptance/D1_6_dual_gpu_validation/gpu_monitor.jsonl`
- `output/carla_acceptance/D1_6_dual_gpu_validation/process_gpu_map.json`
- `output/carla_acceptance/D1_6_dual_gpu_validation/gpu_assignment_timeline.csv`
