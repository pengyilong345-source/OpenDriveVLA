#!/usr/bin/env bash
set -euo pipefail

# Collect five independent episodes for one scenario profile, then validate each.
# Run from the OpenDriveVLA repository root inside WSL.

SCENARIO="${1:-all}"
EPISODES="${EPISODES:-5}"
SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-5}"
RUN_TAG="${RUN_TAG:-small}"
START_EPISODE="${START_EPISODE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

collect_group() {
  local name="$1"
  local config="$2"
  local vehicles="$3"
  local walkers="$4"
  local motorcycles="$5"
  local bicycles="$6"
  local seed_base="$7"
  local nearby_radius="$8"
  local walker_front_fraction="$9"

  echo "==> Collecting ${name}: ${EPISODES} episodes x ${SAMPLES_PER_EPISODE} samples"
  local end_episode=$((START_EPISODE + EPISODES - 1))
  for ((episode = START_EPISODE; episode <= end_episode; episode++)); do
    local seed=$((seed_base + episode))
    local suffix
    suffix=$(printf "%03d" "${episode}")
    local output
    if [[ -n "${OUTPUT_ROOT}" ]]; then
      output="${OUTPUT_ROOT}/episode_${suffix}"
      mkdir -p "${OUTPUT_ROOT}"
    else
      output="carla/output/v1_1_${name}_${RUN_TAG}_${suffix}"
    fi

    echo "--> ${name} episode ${suffix}, seed=${seed}"
    python carla/collectors/multimodal_collect.py \
      --config "${config}" \
      --samples "${SAMPLES_PER_EPISODE}" \
      --vehicles "${vehicles}" \
      --walkers "${walkers}" \
      --motorcycles "${motorcycles}" \
      --bicycles "${bicycles}" \
      --nearby-radius "${nearby_radius}" \
      --walker-front-fraction "${walker_front_fraction}" \
      --seed "${seed}" \
      --output "${output}"

    python carla/collectors/validate_sample_v1_1.py \
      "${output}" \
      --expected-samples "${SAMPLES_PER_EPISODE}"

    if [[ -n "${OUTPUT_ROOT}" ]]; then
      if [[ ! -f "${OUTPUT_ROOT}/progress.tsv" ]]; then
        printf "episode\tseed\tsamples\tstatus\n" > "${OUTPUT_ROOT}/progress.tsv"
      fi
      printf "%s\t%s\t%s\tPASS\n" "${suffix}" "${seed}" "${SAMPLES_PER_EPISODE}" >> "${OUTPUT_ROOT}/progress.tsv"
    fi
  done
}

case "${SCENARIO}" in
  basic)
    collect_group basic carla/scenarios/basic_control.pilot.json 8 3 0 0 1100 60 0.5
    ;;
  complex)
    collect_group complex carla/scenarios/complex_obstacle_avoidance.pilot.json 30 60 5 5 2200 100 0.75
    ;;
  emergency)
    collect_group emergency carla/scenarios/extreme_emergency.pilot.json 12 5 0 0 3300 60 0.6
    ;;
  all)
    collect_group basic carla/scenarios/basic_control.pilot.json 8 3 0 0 1100 60 0.5
    collect_group complex carla/scenarios/complex_obstacle_avoidance.pilot.json 30 60 5 5 2200 100 0.75
    collect_group emergency carla/scenarios/extreme_emergency.pilot.json 12 5 0 0 3300 60 0.6
    ;;
  *)
    echo "Usage: bash carla/collectors/collect_small_scale.sh [basic|complex|emergency|all]" >&2
    exit 2
    ;;
esac

echo "==> Small-scale collection finished successfully."
