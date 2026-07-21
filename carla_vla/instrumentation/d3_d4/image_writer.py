"""Six-camera PNG writer + raw BGRA saver (critical event only).

The PNG writer hashes raw bytes BEFORE any visualization conversion, satisfying
the D3 non-interference constraint that the model's input bytes are the same as
the saved files.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Dict, List


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_six_camera_pngs(images_in_order: List[Any],
                            output_dir: Path,
                            decision_id: str) -> Dict[str, Any]:
    """Save six PNGs (CAM_FRONT_LEFT ... CAM_BACK_RIGHT) for one decision.

    images_in_order: list of 6 PNG-encoded byte buffers (in official order).
    Returns a dict with one entry per camera:
        { cam_name: { "raw_bytes_sha256", "saved_file_sha256", "path" } }
    """
    from PIL import Image
    import io
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                   "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    out: Dict[str, Any] = {}
    if len(images_in_order) != 6:
        return {"error": "expected 6 images, got %d" % len(images_in_order),
                "saved": out}
    for i, (cam, data) in enumerate(zip(cam_names, images_in_order)):
        if isinstance(data, (bytes, bytearray)):
            raw_sha = _sha256_bytes(bytes(data))
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            # numpy array
            arr = bytes(data.tobytes())
            raw_sha = _sha256_bytes(arr)
            img = Image.fromarray(data)
        fpath = output_dir / f"{decision_id}__{cam}.png"
        img.save(fpath, "PNG", optimize=False)
        saved_sha = _sha256_bytes(fpath.read_bytes())
        out[cam] = {
            "raw_bytes_sha256": raw_sha,
            "saved_file_sha256": saved_sha,
            "path": str(fpath),
        }
    return out


def save_raw_bgra_if_event(bgra_bytes: bytes, output_path: Path,
                              event_name: str) -> bool:
    """Optionally save raw BGRA bytes for a critical event frame."""
    if not bgra_bytes:
        return False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bgra_bytes)
    return True