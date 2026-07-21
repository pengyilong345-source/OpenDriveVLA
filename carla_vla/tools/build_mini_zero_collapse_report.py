#!/usr/bin/env python3
"""Aggregate nuScenes-mini zero-collapse diagnostic outputs into comparison JSON.

This is a pure JSON-aggregation tool (no model). It reads:
- the existing baseline 8-sample inference JSON,
- the official-compatible prompt-ablation run,
- (optional) stateful/stateless temporal-audit runs,
- (optional) the perturbation diagnostic run,

and writes structured comparison files:
- output/nuscenes_mini_drivevla/prompt_ablation_comparison.json
- output/nuscenes_mini_drivevla/temporal_mode_comparison.json

It never overwrites baseline JSONs. It computes zero-rates, path lengths, and
per-token transitions only from already-recorded outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def _path_length(traj):
    if not traj or len(traj) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(traj[:-1], traj[1:]):
        total += float(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
    return total


def _is_zero(traj):
    return bool(traj) and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in traj)


def _normalize(rec):
    """Ensure every record has predicted_path_length_m / is_all_zero_trajectory."""
    if rec is None:
        return rec
    traj = rec.get("parsed_trajectory")
    if rec.get("predicted_path_length_m") is None and traj is not None:
        rec["predicted_path_length_m"] = _path_length(traj)
    if rec.get("is_all_zero_trajectory") is None and traj is not None:
        rec["is_all_zero_trajectory"] = _is_zero(traj)
    return rec


def load_json(path):
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def stats(records):
    if not records:
        return None
    zero = sum(1 for r in records if r.get("is_all_zero_trajectory"))
    parse_ok = sum(1 for r in records if r.get("parse_success"))
    lengths = [r.get("predicted_path_length_m") for r in records
               if r.get("predicted_path_length_m") is not None]
    return {
        "sample_count": len(records),
        "parse_success_count": parse_ok,
        "parse_success_rate": parse_ok / len(records),
        "all_zero_count": zero,
        "all_zero_rate": zero / len(records),
        "average_predicted_path_length_m": (float(np.mean(lengths)) if lengths else None),
    }


def per_token_transition(baseline_by_tok, other_by_tok):
    rows = []
    for tok, b in baseline_by_tok.items():
        o = other_by_tok.get(tok)
        if o is None:
            continue
        rows.append({
            "token": tok,
            "baseline_zero": b.get("is_all_zero_trajectory"),
            "other_zero": o.get("is_all_zero_trajectory"),
            "transition": _transition(b.get("is_all_zero_trajectory"), o.get("is_all_zero_trajectory")),
            "baseline_path_length_m": b.get("predicted_path_length_m"),
            "other_path_length_m": o.get("predicted_path_length_m"),
            "baseline_raw_output": b.get("raw_output"),
            "other_raw_output": o.get("raw_output"),
        })
    return rows


def _transition(b_zero, o_zero):
    if b_zero and not o_zero:
        return "zero_to_nonzero"
    if not b_zero and o_zero:
        return "nonzero_to_zero"
    if b_zero and o_zero:
        return "stayed_zero"
    return "stayed_nonzero"


def index_by_token(payload):
    if payload is None:
        return {}
    return {r["token"]: _normalize(r) for r in payload.get("results", [])}


def build_prompt_comparison(args):
    baseline = load_json(args.baseline)
    official = load_json(args.official)
    args.output = args.prompt_comparison_out
    if baseline is None or official is None:
        print("[skip] prompt comparison: missing inputs")
        return
    b_idx = index_by_token(baseline)
    o_idx = index_by_token(official)
    transitions = per_token_transition(b_idx, o_idx)
    payload = {
        "comparison": "baseline_current_mini_vs_official_compatible_mini",
        "baseline_source": str(args.baseline),
        "official_source": str(args.official),
        "baseline_stats": stats(list(b_idx.values())),
        "official_compatible_stats": stats(list(o_idx.values())),
        "transition_summary": {
            "zero_to_nonzero": sum(1 for t in transitions if t["transition"] == "zero_to_nonzero"),
            "nonzero_to_zero": sum(1 for t in transitions if t["transition"] == "nonzero_to_zero"),
            "stayed_zero": sum(1 for t in transitions if t["transition"] == "stayed_zero"),
            "stayed_nonzero": sum(1 for t in transitions if t["transition"] == "stayed_nonzero"),
        },
        "per_token": transitions,
        "field_diffs_shared": (official["results"][0]["field_diff"]
                               if official.get("results") else None),
        "note": ("Only the prompt body fields differ; conversation shell, "
                 "special-token layout, images, camera order, can_bus, "
                 "generation config, and seed are identical."),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote prompt ablation comparison to {args.output}")


def build_temporal_comparison(args):
    stateful = load_json(args.stateful)
    stateless = load_json(args.stateless)
    args.output = args.temporal_comparison_out
    if stateful is None or stateless is None:
        print("[skip] temporal comparison: missing inputs")
        return
    s_idx = index_by_token(stateful)
    l_idx = index_by_token(stateless)
    transitions = per_token_transition(s_idx, l_idx)
    # Did the single non-zero token correlate with history warm-up?
    nonzero_correlation = None
    for tok, rec in s_idx.items():
        if not rec.get("is_all_zero_trajectory"):
            nonzero_correlation = {
                "token": tok, "frame_idx": rec.get("frame_idx"),
                "history_warmup_frames": rec.get("history_warmup_frames"),
                "had_prev_bev_before_generate": (rec.get("temporal_state_before_generate", {})
                                                 .get("prev_bev_available")),
            }
    payload = {
        "comparison": "stateful_vs_stateless_temporal_official_compatible_prompt",
        "stateful_source": str(args.stateful),
        "stateless_source": str(args.stateless),
        "stateful_stats": stats(list(s_idx.values())),
        "stateless_stats": stats(list(l_idx.values())),
        "transition_summary": {
            "zero_to_nonzero": sum(1 for t in transitions if t["transition"] == "zero_to_nonzero"),
            "nonzero_to_zero": sum(1 for t in transitions if t["transition"] == "nonzero_to_zero"),
            "stayed_zero": sum(1 for t in transitions if t["transition"] == "stayed_zero"),
            "stayed_nonzero": sum(1 for t in transitions if t["transition"] == "stayed_nonzero"),
        },
        "nonzero_token_history_correlation": nonzero_correlation,
        "per_token": transitions,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote temporal comparison to {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path,
                   default=Path("output/nuscenes_mini_drivevla/inference_8samples.json"))
    p.add_argument("--official", type=Path,
                   default=Path("output/nuscenes_mini_drivevla/official_prompt_8samples.json"))
    p.add_argument("--prompt-comparison-out", type=Path,
                   default=Path("output/nuscenes_mini_drivevla/prompt_ablation_comparison.json"))
    p.add_argument("--stateful", type=Path,
                   default=Path("output/nuscenes_mini_drivevla/temporal_stateful_8samples.json"))
    p.add_argument("--stateless", type=Path,
                   default=Path("output/nuscenes_mini_drivevla/temporal_stateless_8samples.json"))
    p.add_argument("--temporal-comparison-out", type=Path,
                   default=Path("output/nuscenes_mini_drivevla/temporal_mode_comparison.json"))
    args = p.parse_args()
    build_prompt_comparison(args)
    build_temporal_comparison(args)


if __name__ == "__main__":
    main()
