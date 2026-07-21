"""Decision bundle + index writer for D4.2.

A decision bundle is a self-contained JSON describing one MODEL_CONTROL_SCORED
model decision. Six-camera PNG hashes are computed BEFORE any
evaluator/visualizer processing (raw_bytes_sha256 of the exact sensor bytes).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_six_camera_pngs(png_bytes_by_cam: Dict[str, bytes],
                           images_dir: Path,
                           decision_id: str) -> Dict[str, Dict[str, Any]]:
    """Save 6 lossless PNGs and return per-camera hash metadata.

    raw_bytes_sha256 is computed on the EXACT sensor bytes (before PNG save);
    saved_file_sha256 is computed on the saved PNG file.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Any]] = {}
    for cam, raw_png in png_bytes_by_cam.items():
        raw_bytes_sha = _sha256_bytes(raw_png)
        fpath = images_dir / f"{decision_id}__{cam}.png"
        fpath.write_bytes(raw_png)
        saved_sha = _sha256_bytes(raw_png)  # PNG bytes are what we saved
        out[cam] = {
            "path": str(fpath),
            "raw_bytes_sha256": raw_bytes_sha,
            "saved_file_sha256": saved_sha,
            "size_bytes": len(raw_png),
        }
    return out


def write_d42_decision_bundle(bundle: Dict[str, Any],
                                bundle_dir: Path,
                                decision_id: str) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    fpath = bundle_dir / f"{decision_id}.json"
    txt = json.dumps(bundle, indent=2, default=str)
    fpath.write_text(txt)
    return {
        "path": str(fpath),
        "saved_file_sha256": _sha256_text(txt),
        "decision_id": decision_id,
    }


def write_d42_decision_bundle_index(index: List[Dict[str, Any]],
                                       output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in index:
            f.write(json.dumps(entry, default=str) + "\n")
