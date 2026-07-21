"""Decision bundle writer + global index.

A decision bundle is a single self-contained JSON describing one
MODEL_CONTROL_SCORED frame. The bundle references six PNGs, hashes, prompt,
state, raw output, parsed trajectory, and synchronization metadata.
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


def write_decision_bundle(bundle: Dict[str, Any],
                              bundle_dir: Path,
                              decision_id: str) -> Dict[str, Any]:
    """Persist one decision bundle to disk and return the saved-file hash."""
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


def write_decision_bundle_index(index: List[Dict[str, Any]],
                                    output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in index:
            f.write(json.dumps(entry, default=str) + "\n")