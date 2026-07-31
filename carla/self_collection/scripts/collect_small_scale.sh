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
INTER_EPISODE_DELAY_SECONDS="${INTER_EPISODE_DELAY_SECONDS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TOWN_OVERRIDE="${TOWN_OVERRIDE:-}"
WEATHER_OVERRIDE="${WEATHER_OVERRIDE:-}"
COLLECTOR_TIMEOUT="${COLLECTOR_TIMEOUT:-30}"
MAP_LOAD_SETTLE_SECONDS="${MAP_LOAD_SETTLE_SECONDS:-0}"

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
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}/episode_manifest.json" ]]; then
      echo "    Existing completed episode found; validate and skip: ${output}"
      python carla/self_collection/collectors/validate_sample_v1_1.py \
        "${output}" \
        --expected-samples "${SAMPLES_PER_EPISODE}"
      continue
    fi
    if [[ -d "${output}" && -n "$(ls -A "${output}" 2>/dev/null)" ]]; then
      echo "Incomplete or unrecognized non-empty episode directory: ${output}" >&2
      echo "Move it aside or choose another START_EPISODE/OUTPUT_ROOT; no files were overwritten." >&2
      exit 1
    fi
    local override_args=()
    if [[ -n "${TOWN_OVERRIDE}" ]]; then
      override_args+=(--town "${TOWN_OVERRIDE}")
    fi
    if [[ -n "${WEATHER_OVERRIDE}" ]]; then
      override_args+=(--weather "${WEATHER_OVERRIDE}")
    fi
    python carla/self_collection/collectors/multimodal_collect.py \
      --config "${config}" \
      "${override_args[@]}" \
      --timeout "${COLLECTOR_TIMEOUT}" \
      --map-load-settle-seconds "${MAP_LOAD_SETTLE_SECONDS}" \
      --samples "${SAMPLES_PER_EPISODE}" \
      --vehicles "${vehicles}" \
      --walkers "${walkers}" \
      --motorcycles "${motorcycles}" \
      --bicycles "${bicycles}" \
      --nearby-radius "${nearby_radius}" \
      --walker-front-fraction "${walker_front_fraction}" \
      --seed "${seed}" \
      --output "${output}"

    python carla/self_collection/collectors/validate_sample_v1_1.py \
      "${output}" \
      --expected-samples "${SAMPLES_PER_EPISODE}"

    if [[ -n "${OUTPUT_ROOT}" ]]; then
      if [[ ! -f "${OUTPUT_ROOT}/progress.tsv" ]]; then
        printf "episode\tseed\tsamples\tstatus\n" > "${OUTPUT_ROOT}/progress.tsv"
      fi
      printf "%s\t%s\t%s\tPASS\n" "${suffix}" "${seed}" "${SAMPLES_PER_EPISODE}" >> "${OUTPUT_ROOT}/progress.tsv"
    fi
    sleep "${INTER_EPISODE_DELAY_SECONDS}"
  done
}

case "${SCENARIO}" in
  basic)
    collect_group basic carla/self_collection/scenarios/basic_control.pilot.json 8 3 0 0 1100 60 0.5
    ;;
  basic_40)
    collect_group basic_40 carla/self_collection/scenarios/basic_cruise_40.pilot.json 8 3 0 0 5100 60 0.5
    ;;
  basic_60)
    collect_group basic_60 carla/self_collection/scenarios/basic_accelerate_60.pilot.json 8 0 0 0 6100 60 0.0
    ;;
  basic_stop_restart)
    collect_group basic_stop_restart carla/self_collection/scenarios/basic_stop_restart.pilot.json 8 2 0 0 6200 60 0.3
    ;;
  basic_turn_left)
    collect_group basic_turn_left carla/self_collection/scenarios/basic_turn_left.pilot.json 8 2 0 0 6300 70 0.3
    ;;
  basic_turn_right)
    collect_group basic_turn_right carla/self_collection/scenarios/basic_turn_right.pilot.json 8 2 0 0 6400 70 0.3
    ;;
  basic_lane_change)
    collect_group basic_lane_change carla/self_collection/scenarios/basic_lane_change.pilot.json 8 2 0 0 6500 70 0.3
    ;;
  complex)
    collect_group complex carla/self_collection/scenarios/complex_obstacle_avoidance.pilot.json 24 12 3 3 2200 55 0.7
    ;;
  complex_pedestrian)
    collect_group complex_pedestrian carla/self_collection/scenarios/complex_pedestrian_crossing.pilot.json 12 4 0 0 2300 80 0.5
    ;;
  complex_two_wheeler)
    collect_group complex_two_wheeler carla/self_collection/scenarios/complex_two_wheeler_flow.pilot.json 12 2 4 4 2400 90 0.4
    ;;
  complex_intersection)
    collect_group complex_intersection carla/self_collection/scenarios/complex_intersection_traffic.pilot.json 18 6 2 2 2500 90 0.5
    ;;
  complex_static_obstacle)
    collect_group complex_static_obstacle carla/self_collection/scenarios/complex_static_obstacle_avoidance.pilot.json 12 3 0 0 2600 80 0.4
    ;;
  emergency)
    collect_group emergency carla/self_collection/scenarios/extreme_emergency.pilot.json 10 2 0 0 3300 60 0.5
    ;;
  emergency_brake)
    collect_group emergency_brake carla/self_collection/scenarios/extreme_emergency.pilot.json 10 2 0 0 3300 60 0.5
    ;;
  emergency_cutin)
    collect_group emergency_cutin carla/self_collection/scenarios/extreme_cut_in.pilot.json 10 2 0 0 4400 60 0.5
    ;;
  extreme_e1)
    collect_group extreme_e1 carla/self_collection/scenarios/extreme_e1_cut_in.pilot.json 10 2 0 0 7100 70 0.5
    ;;
  extreme_e2)
    collect_group extreme_e2 carla/self_collection/scenarios/extreme_e2_pedestrian_crossing.pilot.json 8 1 0 0 7200 70 0.2
    ;;
  extreme_e2_critical)
    collect_group extreme_e2_critical carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_critical.pilot.json 8 1 0 0 7250 70 0.2
    ;;
  extreme_e2_hard)
    collect_group extreme_e2_hard carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_hard.pilot.json 8 1 0 0 7230 70 0.2
    ;;
  extreme_e3)
    collect_group extreme_e3 carla/self_collection/scenarios/extreme_e3_construction_merge.pilot.json 8 1 0 0 7300 70 0.2
    ;;
  extreme_e4)
    collect_group extreme_e4 carla/self_collection/scenarios/extreme_e4_lead_hard_brake.pilot.json 10 2 0 0 7400 70 0.5
    ;;
  extreme_all)
    collect_group extreme_e1 carla/self_collection/scenarios/extreme_e1_cut_in.pilot.json 10 2 0 0 7100 70 0.5
    collect_group extreme_e2 carla/self_collection/scenarios/extreme_e2_pedestrian_crossing.pilot.json 8 1 0 0 7200 70 0.2
    collect_group extreme_e2_hard carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_hard.pilot.json 8 1 0 0 7230 70 0.2
    collect_group extreme_e2_critical carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_critical.pilot.json 8 1 0 0 7250 70 0.2
    collect_group extreme_e3 carla/self_collection/scenarios/extreme_e3_construction_merge.pilot.json 8 1 0 0 7300 70 0.2
    collect_group extreme_e4 carla/self_collection/scenarios/extreme_e4_lead_hard_brake.pilot.json 10 2 0 0 7400 70 0.5
    ;;
  taxonomy_all)
    collect_group basic carla/self_collection/scenarios/basic_control.pilot.json 8 3 0 0 1100 60 0.5
    collect_group basic_40 carla/self_collection/scenarios/basic_cruise_40.pilot.json 8 3 0 0 5100 60 0.5
    collect_group basic_60 carla/self_collection/scenarios/basic_accelerate_60.pilot.json 8 0 0 0 6100 60 0.0
    collect_group basic_stop_restart carla/self_collection/scenarios/basic_stop_restart.pilot.json 8 2 0 0 6200 60 0.3
    collect_group basic_turn_left carla/self_collection/scenarios/basic_turn_left.pilot.json 8 2 0 0 6300 70 0.3
    collect_group basic_turn_right carla/self_collection/scenarios/basic_turn_right.pilot.json 8 2 0 0 6400 70 0.3
    collect_group basic_lane_change carla/self_collection/scenarios/basic_lane_change.pilot.json 8 2 0 0 6500 70 0.3
    collect_group complex carla/self_collection/scenarios/complex_obstacle_avoidance.pilot.json 24 12 3 3 2200 55 0.7
    collect_group complex_pedestrian carla/self_collection/scenarios/complex_pedestrian_crossing.pilot.json 12 4 0 0 2300 80 0.5
    collect_group complex_two_wheeler carla/self_collection/scenarios/complex_two_wheeler_flow.pilot.json 12 2 4 4 2400 90 0.4
    collect_group complex_intersection carla/self_collection/scenarios/complex_intersection_traffic.pilot.json 18 6 2 2 2500 90 0.5
    collect_group complex_static_obstacle carla/self_collection/scenarios/complex_static_obstacle_avoidance.pilot.json 12 3 0 0 2600 80 0.4
    collect_group extreme_e1 carla/self_collection/scenarios/extreme_e1_cut_in.pilot.json 10 2 0 0 7100 70 0.5
    collect_group extreme_e2 carla/self_collection/scenarios/extreme_e2_pedestrian_crossing.pilot.json 8 1 0 0 7200 70 0.2
    collect_group extreme_e2_hard carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_hard.pilot.json 8 1 0 0 7230 70 0.2
    collect_group extreme_e2_critical carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_critical.pilot.json 8 1 0 0 7250 70 0.2
    collect_group extreme_e3 carla/self_collection/scenarios/extreme_e3_construction_merge.pilot.json 8 1 0 0 7300 70 0.2
    collect_group extreme_e4 carla/self_collection/scenarios/extreme_e4_lead_hard_brake.pilot.json 10 2 0 0 7400 70 0.5
    ;;
  all)
    collect_group basic carla/self_collection/scenarios/basic_control.pilot.json 8 3 0 0 1100 60 0.5
    collect_group complex carla/self_collection/scenarios/complex_obstacle_avoidance.pilot.json 24 12 3 3 2200 55 0.7
    collect_group emergency_brake carla/self_collection/scenarios/extreme_emergency.pilot.json 10 2 0 0 3300 60 0.5
    collect_group emergency_cutin carla/self_collection/scenarios/extreme_cut_in.pilot.json 10 2 0 0 4400 60 0.5
    ;;
  *)
    echo "Usage: bash carla/self_collection/scripts/collect_small_scale.sh [basic|basic_40|basic_60|basic_stop_restart|basic_turn_left|basic_turn_right|basic_lane_change|complex|complex_pedestrian|complex_two_wheeler|complex_intersection|complex_static_obstacle|extreme_e1|extreme_e2|extreme_e2_hard|extreme_e2_critical|extreme_e3|extreme_e4|extreme_all|taxonomy_all|all]" >&2
    exit 2
    ;;
esac

echo "==> Small-scale collection finished successfully."
