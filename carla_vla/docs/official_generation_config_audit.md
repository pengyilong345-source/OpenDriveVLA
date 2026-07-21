# Official generation-config parity audit (Task 5)

This audit compares the official OpenDriveVLA inference generation settings
(`drivevla/inference_drivevla.py`) against (a) the existing mini baseline
runner (`carla_vla/tools/inference_nuscenes_mini_drivevla.py`) and (b) the
new diagnostic runner (`carla_vla/tools/diag_nuscenes_mini_zero_collapse.py`).

## Official settings

The official path is `inference_drivevilla.py::inference_data`:

```python
cont = model_engine.generate(
    input_ids,
    uniad_data=uniad_data,
    uniad_pth=uniad_pth,
    qa_instance_ind=qa_instance_ind,
    do_sample=False,
    temperature=0,
    max_new_tokens=512,
    num_beams=1,
)
answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
```

There is no `top_p`, `top_k`, `repetition_penalty`, `length_penalty`,
`no_repeat_ngram_size`, custom `stopping_criteria`, or explicit
`eos_token_id`/`pad_token_id` passed. The commented-out block in
`inference_data` (multi-modal trajectory generation with sampling,
`temperature=0.1`, `num_return_sequences=4`) is **not** used by official
inference — it is disabled.

`model_engine.generate` resolves to `LlavaQwenForCausalLM.generate`
(`llava/model/language_model/llava_qwen.py:126`), which forwards the
remaining kwargs to `Qwen2ForCausalLM.generate(super().generate(...))`.
No custom `LogitsProcessor` or `StoppingCriteria` are registered.

Decoding is `tokenizer.batch_decode(cont, skip_special_tokens=True)`.

## Determinism

- `do_sample=False`, `temperature=0`, `num_beams=1` is fully deterministic
  greedy decoding. With a fixed prompt and fixed `inputs_embeds`, the output
  is reproducible regardless of seed. No `torch.manual_seed` is required for
  reproducibility, and official inference sets none.
- The vision tower (`vision_tower_test_mode=True`) and all heads are in
  `eval()` (`model_engine.eval()`); dropout is disabled.

## Parity table

| setting | official | mini baseline runner | diagnostic runner | match? |
|---|---|---|---|---|
| `do_sample` | `False` | `False` | `False` | yes |
| `temperature` | `0` | `0` | `0` | yes |
| `num_beams` | `1` | `1` | `1` | yes |
| `max_new_tokens` | `512` | `64` | `512` (default; `64` available) | **differs (baseline only)** |
| `top_p` | unset | unset | unset | yes |
| `top_k` | unset | unset | unset | yes |
| `repetition_penalty` | unset | unset | unset | yes |
| `length_penalty` | unset | unset | unset | yes |
| `stopping_criteria` | none | none | none | yes |
| `eos_token_id` | unset (model default) | unset | unset | yes |
| `pad_token_id` | unset | unset | unset | yes |
| `skip_special_tokens` (decode) | `True` | `True` | `True` | yes |
| deterministic seed handling | none needed (greedy) | none | none | yes |

## `max_new_tokens` assessment

The only material difference is the official `512` vs the mini baseline's
`64`. A 6-waypoint trajectory token string
`<traj_start>[(x1,y1),...,(x6,y6)]<traj_end>` is well under 64 tokens even
with 2-decimal coordinates, so `64` is sufficient to emit a complete
trajectory. The all-zero collapse is therefore **not** explained by token
budget truncation. To guarantee parity, the diagnostic runner defaults to
`512` and the prompt-ablation run uses `512` as well.

## What this rules out

Generation configuration does not explain the zero collapse: the official
greedy settings are already used (modulo the harmless `max_new_tokens`
difference). Introducing sampling (`do_sample=True`, `temperature>0`) to
force non-zero outputs would violate the non-negotiable constraint against
non-official random sampling as a mitigation, and is not done.

## Conclusion

Decoding is at official parity. The diagnostic runner uses
`do_sample=False, temperature=0, num_beams=1, max_new_tokens=512`,
`skip_special_tokens=True` decoding, and no custom stopping criteria —
identical to official inference. Any zero-output difference between modes
therefore comes from inputs (prompt text, temporal state, image/can_bus),
not from the decoder.
