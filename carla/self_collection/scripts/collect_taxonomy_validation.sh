#!/usr/bin/env bash
set -euo pipefail

# Collect one formally validated 10-sample episode for every taxonomy subtype.
# Run this script from the OpenDriveVLA repository root inside WSL.
#
# Completed episodes are validated and skipped on reruns. Existing files are
# never overwritten.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"

BATCH_ROOT="${BATCH_ROOT:-${WORKSPACE_ROOT}/data/carla_self_collection/taxonomy_latest_validation_20260726}"
START_EPISODE="${START_EPISODE:-1}"
EPISODES="${EPISODES:-1}"
SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-10}"
INTER_EPISODE_DELAY_SECONDS="${INTER_EPISODE_DELAY_SECONDS:-1}"
START_FROM="${START_FROM:-B1_cruise_30}"
END_AT="${END_AT:-E4_lead_hard_brake}"

selection_active=0
selection_started=0
selection_finished=0
selected_count=0

cd "${REPO_ROOT}"

run_subtype() {
  local folder="$1"
  local selector="$2"

  if [[ "${folder}" == "${START_FROM}" ]]; then
    selection_active=1
    selection_started=1
  fi
  if [[ "${selection_active}" != "1" ]]; then
    return
  fi

  echo
  echo "============================================================"
  echo "Formal taxonomy validation: ${folder}"
  echo "Output: ${BATCH_ROOT}/${folder}"
  echo "============================================================"

  OUTPUT_ROOT="${BATCH_ROOT}/${folder}" \
  START_EPISODE="${START_EPISODE}" \
  EPISODES="${EPISODES}" \
  SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE}" \
  INTER_EPISODE_DELAY_SECONDS="${INTER_EPISODE_DELAY_SECONDS}" \
  SKIP_EXISTING=1 \
  RUN_TAG="taxonomy_latest_validation" \
    bash carla/self_collection/scripts/collect_small_scale.sh "${selector}"

  selected_count=$((selected_count + 1))
  if [[ "${folder}" == "${END_AT}" ]]; then
    selection_active=0
    selection_finished=1
  fi
}

mkdir -p "${BATCH_ROOT}"

# B: basic operation (7)
run_subtype "B1_cruise_30" "basic"
run_subtype "B2_cruise_40" "basic_40"
run_subtype "B3_accelerate_60" "basic_60"
run_subtype "B4_stop_restart" "basic_stop_restart"
run_subtype "B5_turn_left" "basic_turn_left"
run_subtype "B6_turn_right" "basic_turn_right"
run_subtype "B7_lane_change" "basic_lane_change"

# C: complex interaction (5)
run_subtype "C1_congested_mixed_traffic" "complex"
run_subtype "C2_pedestrian_crossing" "complex_pedestrian"
run_subtype "C3_two_wheeler_flow" "complex_two_wheeler"
run_subtype "C4_intersection_traffic" "complex_intersection"
run_subtype "C5_static_obstacle" "complex_static_obstacle"

# E: extreme emergency (6)
run_subtype "E1_vehicle_cut_in" "extreme_e1"
run_subtype "E2_pedestrian_crossing" "extreme_e2"
run_subtype "E2_hard_pedestrian" "extreme_e2_hard"
run_subtype "E2_critical_pedestrian" "extreme_e2_critical"
run_subtype "E3_construction_merge" "extreme_e3"
run_subtype "E4_lead_hard_brake" "extreme_e4"

if [[ "${selection_started}" != "1" ]]; then
  echo "Unknown START_FROM subtype: ${START_FROM}" >&2
  exit 2
fi
if [[ "${selection_finished}" != "1" ]]; then
  echo "Unknown or unreachable END_AT subtype: ${END_AT}" >&2
  exit 2
fi

echo
echo "============================================================"
echo "PASS: ${selected_count} selected taxonomy subtype(s) collected and validated."
echo "Samples: $((selected_count * EPISODES * SAMPLES_PER_EPISODE))"
echo "Output: ${BATCH_ROOT}"
echo "============================================================"
