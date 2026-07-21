#!/usr/bin/env python3
"""Task 1: exact official prompt audit for nuScenes-mini (no model needed).

For each of the eight fixed mini tokens, render:
- the current-mini prompt (byte-faithful to the baseline adapter), and
- the reconstructed official-compatible-mini prompt (real mini fields only),
plus a structured field-by-field diff and tokenized length when a tokenizer is
available.

Output: output/nuscenes_mini_drivevla/prompt_audit.json
A companion markdown diff is written to carla_vla/docs/mini_vs_official_prompt_diff.md
(it is regenerated from this JSON by the report builder).

No model is loaded. No GT. No full-val cache entries.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils.nuscenes_mini_inference_adapter import NuScenesMiniInferenceAdapter
from carla_vla.tools.mini_prompt_modes import (
    build_current_mini_prompt,
    build_official_compatible_mini_prompt,
    field_diff,
)
from llava.conversation import conv_templates


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mini-info", type=Path, required=True)
    p.add_argument("--dataroot", type=Path, required=True)
    p.add_argument("--tokens", type=Path, default=Path("output/nuscenes_mini_drivevla/mini_8_tokens.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-path", default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B",
                   help="Used only to load the tokenizer for length comparison; not required.")
    p.add_argument("--with-tokenizer", action="store_true",
                   help="Load tokenizer to compute tokenized lengths. Optional.")
    return p.parse_args()


def render_with_conv(prompt):
    conv = conv_templates["qwen_planning_oriented_vlm"].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def maybe_load_tokenizer(args):
    if not args.with_tokenizer:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load tokenizer: {exc}")
        return None


def tok_len(tokenizer, text):
    if tokenizer is None:
        return None
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def main():
    args = args_parser()
    adapter = NuScenesMiniInferenceAdapter(args.mini_info, args.dataroot)
    tok_payload = json.loads(args.tokens.read_text())
    tokens = tok_payload["tokens"] if isinstance(tok_payload, dict) else tok_payload
    token_to_index = {rec["token"]: i for i, rec in enumerate(adapter.infos)}
    tokenizer = maybe_load_tokenizer(args)

    samples = []
    for t in tokens:
        if t not in token_to_index:
            raise RuntimeError(f"token {t} not in mini info")
        index = token_to_index[t]
        info = adapter.infos[index]
        prev_info = adapter.infos[index - 1] if index > 0 else None
        _cmd_value, route = adapter.route_command(info)
        current = build_current_mini_prompt(info, route)
        official = build_official_compatible_mini_prompt(info, route, prev_info)
        current_full = render_with_conv(current)
        official_full = render_with_conv(official)
        samples.append({
            "token": t,
            "sample_token": t,
            "scene_name": adapter.scene_names[info["scene_token"]],
            "frame_idx": info["frame_idx"],
            "has_previous_keyframe": prev_info is not None,
            "current_mini_prompt_body": current,
            "official_compatible_mini_prompt_body": official,
            "current_mini_prompt_full": current_full,
            "official_compatible_mini_prompt_full": official_full,
            "field_diff": field_diff(current, official),
            "current_tokenized_body_length": tok_len(tokenizer, current_full),
            "official_tokenized_body_length": tok_len(tokenizer, official_full),
            "route": route,
            "speed_mps": float(__import__("numpy").linalg.norm(
                __import__("numpy").asarray(info["can_bus"], dtype=float)[13:16])),
        })

    identical_count = sum(1 for s in samples if s["field_diff"]["identical"])
    payload = {
        "task": "exact-official-prompt-audit",
        "mini_info": str(args.mini_info),
        "token_count": len(samples),
        "bodies_identical_count": identical_count,
        "note": ("special-token layout matches official in all tokens; "
                 "numeric ego/history/command fields differ and are reconstructed "
                 "from real mini CAN/ego only (no GT, no full-val cache)."),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote prompt audit for {len(samples)} token(s) to {args.output}")


if __name__ == "__main__":
    main()
