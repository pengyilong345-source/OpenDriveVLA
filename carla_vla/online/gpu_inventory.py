"""D1.6 Phase 0 — GPU + environment audit.

Collects:
  - nvidia-smi GPU inventory (index / UUID / PCI / model / mem / driver)
  - PyTorch + CUDA inventory (per env)
  - CARLA rendering environment (Vulkan ICDs, processes, current GPU)
  - GPU topology
  - Module / library versions

Writes output/carla_acceptance/D1_6_dual_gpu_validation/:
  gpu_inventory.json
  gpu_topology.txt
  environment_snapshot.txt
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "carla_acceptance" / "D1_6_dual_gpu_validation"


def _run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout
    except Exception as e:
        return f"<cmd {cmd!r} failed: {e}>"


def collect() -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1) nvidia-smi GPU inventory
    smi_q = _run("nvidia-smi -q")
    gpus: List[Dict[str, Any]] = []
    for blk in smi_q.split("\nGPU "):
        if not blk.strip():
            continue
        # After split: block 0 has preamble; block 1+ starts with the
        # bus_id "00000000:XX:XX.X". We add "GPU " back so the regex
        # matches consistently. Skip the preamble (block 0).
        if not re.search(r"^00000000:", blk.lstrip()):
            continue
        candidate = "GPU " + blk
        # bus_id like "00000000:65:00.0" appears right after "GPU "
        m = re.search(r"GPU (\d{8}:[\dA-Fa-f]{2}:\d{2}\.\d+)", candidate)
        if not m:
            # fallback: take whatever the next 12-char token is
            m = re.search(r"GPU (\S{12})", candidate)
            if not m:
                continue
        bus_id = m.group(1)
        # parse common fields
        def grab(prefix: str) -> str:
            mm = re.search(rf"{re.escape(prefix)}\s*:\s*(.+?)(?:\n\s|$)", blk, re.DOTALL)
            return mm.group(1).strip() if mm else ""
        gpu = {
            "bus_id": bus_id,
            "product_name": grab("Product Name"),
            "product_brand": grab("Product Brand"),
            "product_architecture": grab("Product Architecture"),
            "uuid": grab("GPU UUID"),
            "vbios_version": grab("VBIOS Version"),
            "minor_number": grab("Minor Number"),
            "display_active": grab("Display Active"),
            "persistence_mode": grab("Persistence Mode"),
            "current_memory_usage_mib":
                (lambda m: re.search(r"(\d+)MiB / ", m).group(1) if m else "")(grab("FB Memory Usage") or ""),
            "raw_block": blk[:4000],
        }
        gpus.append(gpu)
    out["nvidia_smi_gpus"] = gpus
    out["driver_version"] = re.search(r"Driver Version\s*:\s*(\S+)", smi_q).group(1)
    out["cuda_version_runtime"] = re.search(r"CUDA Version\s*:\s*(\S+)", smi_q).group(1)
    out["attached_gpu_count"] = re.search(r"Attached GPUs\s*:\s*(\d+)", smi_q).group(1)

    # 2) PyTorch inventory in base env
    base_torch: Dict[str, Any] = {}
    try:
        base_torch["torch_version"] = _run("bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -c \"import torch; print(torch.__version__)\"'").strip()
    except Exception:
        pass
    try:
        base_torch["cuda_runtime_in_python"] = _run("bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python -c \"import torch; print(torch.version.cuda); print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])\"'")
    except Exception:
        pass
    out["base_env_torch"] = base_torch

    # 3) CARLA rendering environment
    vulkan: Dict[str, Any] = {}
    vulkan["icd_dir"] = _run("ls /etc/vulkan/icd.d/").strip()
    vulkan["my_nvidia_icd"] = _run("cat /etc/vulkan/icd.d/my_nvidia_icd.json").strip()
    vulkan["nvidia_icd"] = _run("cat /etc/vulkan/icd.d/nvidia_icd.json").strip()
    # Find CARLA UE4 process(es) and their physical GPU
    carla_proc = _run("ps -ef | grep -E 'CarlaUE4' | grep -v grep").strip()
    out["carla_processes_ps"] = carla_proc
    # nvidia-smi pmon for CARLA processes
    pmon = _run("nvidia-smi pmon -c 1 -s u")
    out["carla_pmon"] = pmon
    # enumerate compute apps
    apps = _run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv").strip()
    out["carla_compute_apps"] = apps
    out["vulkan"] = vulkan

    # 4) GPU topology
    out["topology"] = _run("nvidia-smi topo -m").strip()

    # 5) Module / library versions
    mods: Dict[str, str] = {}
    for cmd in [
        "python --version", "pip list 2>/dev/null | head -200",
    ]:
        mods[cmd] = _run(cmd).strip()
    out["python_env"] = mods

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    (OUT_DIR / "gpu_inventory.json").write_text(
        json.dumps(data, indent=2, default=str))
    (OUT_DIR / "gpu_topology.txt").write_text(
        data.get("topology", "") + "\n")
    # env_snapshot: free-form text dump
    lines: List[str] = []
    lines.append("=== nvidia-smi -q ===")
    lines.append(_run("nvidia-smi -q"))
    lines.append("=== nvidia-smi topo -m ===")
    lines.append(data.get("topology", ""))
    lines.append("=== Vulkan ICDs ===")
    lines.append(_run("ls /etc/vulkan/icd.d/"))
    lines.append(_run("cat /etc/vulkan/icd.d/my_nvidia_icd.json"))
    lines.append("=== ps -ef for CarlaUE4 ===")
    lines.append(data.get("carla_processes_ps", ""))
    lines.append("=== nvidia-smi pmon -c 1 -s u ===")
    lines.append(data.get("carla_pmon", ""))
    lines.append("=== nvidia-smi compute apps ===")
    lines.append(data.get("carla_compute_apps", ""))
    (OUT_DIR / "environment_snapshot.txt").write_text("\n".join(lines))
    print(f"wrote gpu_inventory.json ({len(data['nvidia_smi_gpus'])} GPUs)")
    print(f"wrote gpu_topology.txt")
    print(f"wrote environment_snapshot.txt")


if __name__ == "__main__":
    main()
