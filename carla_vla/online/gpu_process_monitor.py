"""D1.6 Phase 2 — GPU process monitor (v2 — clean).

Watches nvidia-smi at 100-200 ms cadence. Records per-GPU memory/util
and per-process (compute app) GPU assignment.

Output:
  output/carla_acceptance/D1_6_dual_gpu_validation/gpu_monitor.jsonl
  output/carla_acceptance/D1_6_dual_gpu_validation/gpu_assignment_verification.json
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:
        return ""


def _parse_csv_noheader(output: str):
    """Each line is a CSV row; return list of dicts with first col as 'idx'."""
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        rec = {"idx": parts[0]}
        for i, p in enumerate(parts[1:], start=1):
            rec[f"c{i}"] = p
        rows.append(rec)
    return rows


def _cycle() -> dict:
    out = {
        "t": time.time(),
        "iso": datetime.now().isoformat(timespec="seconds"),
        "per_gpu": {},
        "per_process": [],
    }
    # per-GPU: nvidia-smi --query-gpu=index,... --format=csv,noheader,nounits
    smi_q = _run(
        "nvidia-smi --query-gpu=index,gpu_uuid,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits"
    )
    for r in _parse_csv_noheader(smi_q):
        idx = r["idx"]
        out["per_gpu"][idx] = {
            "uuid": r.get("c1", ""),
            "memory_mib": r.get("c2", "0"),
            "gpu_util_pct": r.get("c3", "0"),
            "mem_util_pct": r.get("c4", "0"),
            "power_w": r.get("c5", "0"),
            "temp_c": r.get("c6", "0"),
        }
    # per-process: nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory
    apps = _run(
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits"
    )
    for r in _parse_csv_noheader(apps):
        out["per_process"].append({
            "gpu_uuid": r.get("idx", ""),  # first col is gpu_uuid here
            "pid": r.get("c1", ""),
            "process_name": r.get("c2", ""),
            "used_memory_mib": r.get("c3", "0"),
        })
    return out


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "n": 0}
    return {
        "min": min(values), "max": max(values), "mean": sum(values) / len(values),
        "n": len(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="output/carla_acceptance/D1_6_dual_gpu_validation/gpu_monitor.jsonl")
    ap.add_argument("--summary",
                    default="output/carla_acceptance/D1_6_dual_gpu_validation/gpu_assignment_verification.json")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds (0 = run forever)")
    ap.add_argument("--cadence", type=float, default=0.2)
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    rows: list = []
    started = time.time()
    # Make the output an absolute path so backgrounding doesn't shift CWD
    args.output = str(Path(args.output).resolve())
    args.summary = str(Path(args.summary).resolve())
    print(f"[gpu-monitor] writing to {args.output} cadence={args.cadence}s duration={args.duration}s",
            flush=True)
    with open(args.output, "w") as f:
        while True:
            if args.duration > 0 and (time.time() - started) >= args.duration:
                break
            try:
                row = _cycle()
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as e:
                print(f"[gpu-monitor] err: {e}", file=sys.stderr)
            time.sleep(args.cadence)
    # Build summary
    by_gpu_mem = defaultdict(list)
    by_gpu_util = defaultdict(list)
    by_gpu_power = defaultdict(list)
    by_gpu_temp = defaultdict(list)
    by_gpu_uuid = {}
    proc_by_gpu = defaultdict(set)
    for r in rows:
        for idx, info in r["per_gpu"].items():
            by_gpu_mem[idx].append(float(info.get("memory_mib", 0) or 0))
            by_gpu_util[idx].append(float(info.get("gpu_util_pct", 0) or 0))
            by_gpu_power[idx].append(float(info.get("power_w", 0) or 0))
            by_gpu_temp[idx].append(float(info.get("temp_c", 0) or 0))
            by_gpu_uuid[idx] = info.get("uuid", "")
        for p in r["per_process"]:
            if p["gpu_uuid"]:
                proc_by_gpu[p["gpu_uuid"]].add(f"{p['process_name']}:{p['pid']}")
    summary = {
        "n_rows": len(rows),
        "duration_s": round(rows[-1]["t"] - rows[0]["t"], 2) if rows else 0.0,
        "per_gpu": {
            idx: {
                "uuid": by_gpu_uuid.get(idx, ""),
                "memory_mib": _stats(by_gpu_mem.get(idx, [])),
                "gpu_util_pct": _stats(by_gpu_util.get(idx, [])),
                "power_w": _stats(by_gpu_power.get(idx, [])),
                "temp_c": _stats(by_gpu_temp.get(idx, [])),
            }
            for idx in by_gpu_uuid
        },
        "processes_seen_per_gpu_uuid": {
            uuid: sorted(procs) for uuid, procs in proc_by_gpu.items()
        },
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[gpu-monitor] wrote {args.output} ({len(rows)} rows) and {args.summary}",
            flush=True)


if __name__ == "__main__":
    main()
