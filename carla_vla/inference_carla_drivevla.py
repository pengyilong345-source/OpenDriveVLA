"""Run OpenDriveVLA text generation on the mock CARLA dataset."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import deepspeed
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils import CAMERA_NAMES, CARLA_DATA_ROOT, CARLA_INFO_PATH, CarlaLLaVADataset, CarlaUniADAdapter  # noqa: E402
from carla_vla.data_utils.carla_uniad_adapter import NATIVE_PLACEHOLDER_MODES  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token, tokenizer_uniad_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


TRAJECTORY_FORMAT_INSTRUCTION = (
    "Output the future ego trajectory exactly in this format:\n"
    "<traj_start>[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]<traj_end>"
)

TRAJECTORY_PATTERN = re.compile(r"<traj_start>\s*(\[.*?\])\s*<traj_end>", re.DOTALL)

NATIVE_GT_PLACEHOLDER_KEYS = [
    "sdc_planning",
    "sdc_planning_mask",
    "gt_lane_labels",
    "gt_lane_masks",
    "gt_segmentation",
    "gt_instance",
    "gt_occ_img_is_valid",
]


def load_model_with_deepspeed(args: argparse.Namespace, device: torch.device):
    """Load OpenDriveVLA with the same model/deepspeed setup as drivevla inference."""
    disable_torch_init()
    ensure_single_process_deepspeed_env()

    llava_model_args = {
        "multimodal": True,
        "attn_implementation": args.attn_implementation,
        "overwrite_config": {
            "image_aspect_ratio": "pad",
            "vision_tower_test_mode": True,
        },
    }

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map=device,
        **llava_model_args,
    )

    ds_config = {
        "fp16": {"enabled": args.fp16},
        "bf16": {"enabled": args.bf16},
        "zero_optimization": {"stage": 0},
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
        "inference_mode": True,
    }

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=[],
    )

    return tokenizer, model_engine, image_processor, context_len


def ensure_single_process_deepspeed_env() -> None:
    """Let DeepSpeed initialize without probing MPI when launched by plain python."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")


def build_carla_generation_prompt(
    dataset_prompt: str,
    use_image_tokens: bool,
    image_summaries: list[str],
) -> str:
    if use_image_tokens:
        image_context = "\n".join(f"<image> {camera_name}" for camera_name in CAMERA_NAMES)
    else:
        image_context = "Camera inputs:\n" + "\n".join(image_summaries)

    question = "\n\n".join([image_context, dataset_prompt, TRAJECTORY_FORMAT_INSTRUCTION])

    conv = conv_templates["qwen_planning_oriented_vlm"].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare_sample_inputs(
    sample: dict[str, Any],
    tokenizer,
    image_processor,
    model_config,
    device: torch.device,
    image_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor] | None, list[tuple[int, int]], str, str]:
    pil_images = [sample["images"][camera_name] for camera_name in CAMERA_NAMES]
    image_sizes = [image.size for image in pil_images]
    image_summaries = [
        f"- {camera_name}: path={sample['image_paths'][camera_name]}, size={sample['images'][camera_name].size}"
        for camera_name in CAMERA_NAMES
    ]

    use_image_tensors = image_processor is not None
    prompt = build_carla_generation_prompt(
        sample["prompt"],
        use_image_tokens=use_image_tensors,
        image_summaries=image_summaries,
    )
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)

    if not use_image_tensors:
        return input_ids, None, image_sizes, prompt, "camera_metadata"

    images = process_images(pil_images, image_processor, model_config)
    if isinstance(images, list):
        images = [image.to(device=device, dtype=image_dtype) for image in images]
    else:
        images = images.to(device=device, dtype=image_dtype)

    return input_ids, images, image_sizes, prompt, "image_tensors"


def generate_text(
    model_engine,
    tokenizer,
    input_ids: torch.Tensor,
    images,
    image_sizes: list[tuple[int, int]],
    args: argparse.Namespace,
) -> str:
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=device_supports_cuda(input_ids.device), dtype=autocast_dtype):
            generation_kwargs = {
                "do_sample": False,
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": 1,
            }
            if images is not None:
                generation_kwargs.update(
                    {
                        "images": images,
                        "image_sizes": image_sizes,
                        "modalities": ["image"],
                    }
                )

            output_ids = model_engine.generate(input_ids, **generation_kwargs)

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def normalize_trajectory(candidate: Any) -> list[list[float]] | None:
    if not isinstance(candidate, list) or len(candidate) != 6:
        return None

    parsed_trajectory = []
    for point in candidate:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        parsed_trajectory.append([float(x), float(y)])
    return parsed_trajectory


def bracket_candidates(text: str) -> list[str]:
    candidates = []
    stack = []
    start = None
    for index, char in enumerate(text):
        if char == "[":
            if not stack:
                start = index
            stack.append(char)
        elif char == "]" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:index + 1])
                start = None
    return candidates


def parse_trajectory(raw_output: str) -> tuple[list[list[float]] | None, bool, str | None]:
    match = TRAJECTORY_PATTERN.search(raw_output)
    if match is not None:
        try:
            trajectory = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            trajectory = None
        parsed = normalize_trajectory(trajectory)
        if parsed is not None:
            return parsed, True, "tagged"

    bare_candidates = [raw_output.strip()]
    bare_candidates.extend(bracket_candidates(raw_output))
    for candidate_text in bare_candidates:
        try:
            trajectory = ast.literal_eval(candidate_text)
        except (SyntaxError, ValueError):
            continue
        parsed = normalize_trajectory(trajectory)
        if parsed is not None:
            return parsed, True, "bare_list"

    return None, False, None


def is_all_zero_trajectory(trajectory: list[list[float]] | None, eps: float = 1e-4) -> bool:
    if trajectory is None:
        return False
    return all(abs(x) <= eps and abs(y) <= eps for x, y in trajectory)


def device_supports_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def move_data_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_data_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(move_data_to_device(value, device) for value in data)
    return data


def tensor_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return f"list[Tensor](shape0={list(value[0].shape)}, dtype0={value[0].dtype}, device0={value[0].device})"
    return type(value).__name__


def debug_uniad_data(uniad_data: dict[str, Any]) -> None:
    img_metas = uniad_data.get("img_metas")
    meta_keys = []
    if isinstance(img_metas, list) and img_metas and isinstance(img_metas[0], list) and img_metas[0]:
        meta_keys = sorted(img_metas[0][0].keys())
    print("[carla_inference] native uniad_data enabled")
    print(f"[carla_inference] img: {tensor_summary(uniad_data.get('img'))}")
    print(f"[carla_inference] cameras: {len(CAMERA_NAMES)}")
    print(f"[carla_inference] img_metas keys: {meta_keys}")
    print(f"[carla_inference] can_bus: {type(img_metas[0][0].get('can_bus')).__name__ if meta_keys else 'missing'}")
    print(f"[carla_inference] lidar2img: {type(img_metas[0][0].get('lidar2img')).__name__ if meta_keys else 'missing'}")


def build_native_generation_prompt(dataset_prompt: str) -> str:
    conv = conv_templates["qwen_planning_oriented_vlm"].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], dataset_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare_native_sample_inputs(
    sample: dict[str, Any],
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], str, str]:
    prompt = build_native_generation_prompt(sample["prompt"])
    input_ids = tokenizer_uniad_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    uniad_data = move_data_to_device(sample["uniad_data"], device)
    return input_ids, uniad_data, prompt, "uniad_data"


def generate_text_with_uniad_data(
    model_engine,
    tokenizer,
    input_ids: torch.Tensor,
    uniad_data: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=device_supports_cuda(input_ids.device), dtype=autocast_dtype):
            output_ids = model_engine.generate(
                input_ids,
                uniad_data=uniad_data,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def run_inference(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "carla_inference_results.json"

    device = torch.device(args.device)
    if args.use_uniad_data:
        print("[carla_inference] --use-uniad-data enabled")
        print(f"[carla_inference] include native GT placeholders: {not args.disable_native_gt_placeholders}")
        print(f"[carla_inference] include route in prompt: {not args.disable_route_in_prompt}")
        print(f"[carla_inference] native placeholder mode: {args.native_placeholder_mode}")
        dataset = CarlaUniADAdapter(
            info_path=args.carla_info,
            data_root=args.carla_data_root,
            load_images=True,
            include_gt_placeholders=not args.disable_native_gt_placeholders,
            include_route_in_prompt=not args.disable_route_in_prompt,
            native_placeholder_mode=args.native_placeholder_mode,
        )
    else:
        print("[carla_inference] prompt-only CARLA inference path enabled")
        print(f"[carla_inference] include route in prompt: {not args.disable_route_in_prompt}")
        dataset = CarlaLLaVADataset(
            info_path=args.carla_info,
            data_root=args.carla_data_root,
            load_images=True,
            include_route_in_prompt=not args.disable_route_in_prompt,
        )

    max_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    tokenizer, model_engine, image_processor, _ = load_model_with_deepspeed(args, device)
    model_engine.eval()
    model_config = model_engine.module.config if hasattr(model_engine, "module") else model_engine.config
    image_dtype = torch.bfloat16 if args.bf16 else torch.float16

    results = []
    for index in tqdm(range(max_samples), ncols=80):
        sample = dataset[index]
        used_uniad_data = False
        native_data_keys = []
        if args.use_uniad_data:
            try:
                input_ids, uniad_data, prompt, image_mode = prepare_native_sample_inputs(
                    sample=sample,
                    tokenizer=tokenizer,
                    device=device,
                )
                native_data_keys = sorted(uniad_data.keys())
                if index == 0:
                    debug_uniad_data(uniad_data)
                raw_output = generate_text_with_uniad_data(
                    model_engine=model_engine,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    uniad_data=uniad_data,
                    args=args,
                )
                used_uniad_data = True
            except Exception as exc:
                if not args.allow_prompt_fallback:
                    missing_native_keys = []
                    if "uniad_data" in locals() and isinstance(uniad_data, dict):
                        missing_native_keys = [key for key in NATIVE_GT_PLACEHOLDER_KEYS if key not in uniad_data]
                    removed_note = (
                        " Removed native GT/planning placeholder keys: {}.".format(missing_native_keys)
                        if missing_native_keys else ""
                    )
                    raise RuntimeError(
                        "Native CARLA uniad_data path failed. Missing/invalid native fields should be fixed before fallback."
                        + removed_note
                        + " Use --allow-prompt-fallback to explicitly fall back. Original error: {}".format(exc)
                    ) from exc
                print("[carla_inference] WARNING: native path failed, falling back to prompt-only: {}".format(exc))
                prompt_sample = CarlaLLaVADataset(
                    args.carla_info,
                    args.carla_data_root,
                    load_images=True,
                    include_route_in_prompt=not args.disable_route_in_prompt,
                )[index]
                input_ids, images, image_sizes, prompt, image_mode = prepare_sample_inputs(
                    sample=prompt_sample,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    model_config=model_config,
                    device=device,
                    image_dtype=image_dtype,
                )
                raw_output = generate_text(
                    model_engine=model_engine,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    args=args,
                )
                sample = prompt_sample
        else:
            input_ids, images, image_sizes, prompt, image_mode = prepare_sample_inputs(
                sample=sample,
                tokenizer=tokenizer,
                image_processor=image_processor,
                model_config=model_config,
                device=device,
                image_dtype=image_dtype,
            )
            raw_output = generate_text(
                model_engine=model_engine,
                tokenizer=tokenizer,
                input_ids=input_ids,
                images=images,
                image_sizes=image_sizes,
                args=args,
            )
        parsed_trajectory, parse_success, parse_format = parse_trajectory(raw_output)
        all_zero = is_all_zero_trajectory(parsed_trajectory)
        trajectory_warning = "all_zero_trajectory" if all_zero else None

        result = {
            "sample_id": sample["sample_id"],
            "timestamp": sample["timestamp"],
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed_trajectory": parsed_trajectory,
            "parse_success": parse_success,
            "parse_format": parse_format,
            "trajectory_warning": trajectory_warning,
            "is_all_zero_trajectory": all_zero,
            "image_paths": {name: str(path) for name, path in sample["image_paths"].items()},
            "image_sizes": {name: list(sample["images"][name].size) for name in CAMERA_NAMES},
            "image_mode": image_mode,
            "ego": sample["ego"],
            "map": sample["map"],
            "agents": sample["agents"],
            "route_waypoints": sample.get("route_waypoints"),
            "used_uniad_data": used_uniad_data,
            "disabled_native_gt_placeholders": bool(args.disable_native_gt_placeholders),
            "disabled_route_in_prompt": bool(args.disable_route_in_prompt),
            "native_placeholder_mode": args.native_placeholder_mode if used_uniad_data else None,
            "native_data_keys": native_data_keys if used_uniad_data else [],
        }

        results.append(result)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "carla_info": str(args.carla_info),
                "carla_data_root": str(args.carla_data_root),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenDriveVLA inference on CARLA-style mock data.")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    parser.add_argument("--carla-info", type=Path, default=CARLA_INFO_PATH)
    parser.add_argument("--carla-data-root", type=Path, default=CARLA_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/carla_drivevla"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation to use.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--use-uniad-data", action="store_true", help="Use CARLA native-like uniad_data instead of prompt-only image tensors.")
    parser.add_argument("--disable-native-gt-placeholders", action="store_true", help="Remove CARLA native-like GT/planning placeholder fields from uniad_data for ablation.")
    parser.add_argument("--native-placeholder-mode", choices=NATIVE_PLACEHOLDER_MODES, default="zero_current", help="Native placeholder value strategy for UniAD ablations.")
    parser.add_argument("--disable-route-in-prompt", action="store_true", help="Do not include route_waypoints in the text prompt for ablation.")
    parser.add_argument("--allow-prompt-fallback", action="store_true", help="Explicitly fall back to prompt-only inference if native uniad_data fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_inference(args)
    print(f">>> CARLA inference results saved to {output_path}")


if __name__ == "__main__":
    main()

