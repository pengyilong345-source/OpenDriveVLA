#!/usr/bin/env bash
set -uo pipefail

# Collect the 12,100-frame emergency portion of production_33k_v1_0.
#
# Design goals:
# - use only the 22 map/scenario combinations qualified in July 2026;
# - save 10 synchronized samples per episode (1,210 episodes total);
# - keep running after an individual episode fails;
# - skip and revalidate completed episodes on restart;
# - preserve incomplete attempts for later inspection;
# - print and save an aggregate PASS/FAIL summary after all selected work.
#
# Optional filters:
#   ONLY_TOWN=Town03
#   ONLY_SUBTYPE=E1_vehicle_cut_in
#   DRY_RUN=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"

PRODUCTION_ROOT="${PRODUCTION_ROOT:-${WORKSPACE_ROOT}/data/carla_self_collection/production_33k_v1_0/emergency}"
SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-10}"
INTER_EPISODE_DELAY_SECONDS="${INTER_EPISODE_DELAY_SECONDS:-2}"
INTER_COMBINATION_DELAY_SECONDS="${INTER_COMBINATION_DELAY_SECONDS:-5}"
COLLECTOR_TIMEOUT="${COLLECTOR_TIMEOUT:-90}"
MAP_LOAD_SETTLE_SECONDS="${MAP_LOAD_SETTLE_SECONDS:-10}"
ONLY_TOWN="${ONLY_TOWN:-}"
ONLY_SUBTYPE="${ONLY_SUBTYPE:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"

if (( SAMPLES_PER_EPISODE != 10 )); then
  echo "This production plan requires exactly 10 samples per episode." >&2
  exit 2
fi

REPORT_ROOT="${PRODUCTION_ROOT}/reports/${RUN_ID}"
SUMMARY_FILE="${REPORT_ROOT}/emergency_12100_summary.tsv"
FAILURE_FILE="${REPORT_ROOT}/failed_episodes.tsv"
PLAN_FILE="${REPORT_ROOT}/selected_plan.tsv"

mkdir -p "${REPORT_ROOT}"
printf "subtype\tselector\ttown\tweather\tstart_episode\tepisodes\tplanned_samples\toutput\n" > "${PLAN_FILE}"
printf "subtype\ttown\tweather\tepisode\tstatus\tsamples\toutput\n" > "${SUMMARY_FILE}"
printf "subtype\ttown\tweather\tepisode\treason\toutput\n" > "${FAILURE_FILE}"

cd "${REPO_ROOT}"

# start_episode is global within each subtype so every map uses distinct seeds.
# Allocation:
# E1 2,200; E2-safe 1,800; E2-hard 1,800; E2-critical 1,600;
# E3 2,100; E4 2,600. Total: 12,100 frames.
read -r -d '' PLAN_MATRIX <<'EOF' || true
E2_safe_pedestrian|extreme_e2|Town01|clear_day|1|45
E2_hard_pedestrian|extreme_e2_hard|Town01|clear_day|1|45
E2_critical_pedestrian|extreme_e2_critical|Town01|clear_day|1|40
E4_lead_hard_brake|extreme_e4|Town01|clear_day|1|65
E1_vehicle_cut_in|extreme_e1|Town03|clear_day|1|74
E2_safe_pedestrian|extreme_e2|Town03|clear_day|46|45
E2_hard_pedestrian|extreme_e2_hard|Town03|clear_day|46|45
E2_critical_pedestrian|extreme_e2_critical|Town03|clear_day|41|40
E3_construction_merge|extreme_e3|Town03|clear_day|1|70
E4_lead_hard_brake|extreme_e4|Town03|clear_day|66|65
E1_vehicle_cut_in|extreme_e1|Town05|clear_day|75|73
E2_safe_pedestrian|extreme_e2|Town05|clear_day|91|45
E2_hard_pedestrian|extreme_e2_hard|Town05|clear_day|91|45
E2_critical_pedestrian|extreme_e2_critical|Town05|clear_day|81|40
E3_construction_merge|extreme_e3|Town05|clear_day|71|70
E4_lead_hard_brake|extreme_e4|Town05|clear_day|131|65
E1_vehicle_cut_in|extreme_e1|Town10HD_Opt|clear_day|148|73
E2_safe_pedestrian|extreme_e2|Town10HD_Opt|clear_day|136|45
E2_hard_pedestrian|extreme_e2_hard|Town10HD_Opt|clear_day|136|45
E2_critical_pedestrian|extreme_e2_critical|Town10HD_Opt|clear_day|121|40
E3_construction_merge|extreme_e3|Town10HD_Opt|clear_day|141|70
E4_lead_hard_brake|extreme_e4|Town10HD_Opt|clear_day|196|65
EOF

selected_combinations=0
planned_episodes=0
planned_samples=0
pass_episodes=0
fail_episodes=0

combination_index=0
while IFS='|' read -r subtype selector town weather start_episode episodes; do
  [[ -z "${subtype}" ]] && continue
  if [[ -n "${ONLY_TOWN}" && "${town}" != "${ONLY_TOWN}" ]]; then
    continue
  fi
  if [[ -n "${ONLY_SUBTYPE}" && "${subtype}" != "${ONLY_SUBTYPE}" ]]; then
    continue
  fi

  selected_combinations=$((selected_combinations + 1))
  planned_episodes=$((planned_episodes + episodes))
  planned_samples=$((planned_samples + episodes * SAMPLES_PER_EPISODE))
  combination_index=$((combination_index + 1))

  output_root="${PRODUCTION_ROOT}/${subtype}/${town}_${weather}"
  end_episode=$((start_episode + episodes - 1))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${subtype}" "${selector}" "${town}" "${weather}" \
    "${start_episode}" "${episodes}" "$((episodes * SAMPLES_PER_EPISODE))" \
    "${output_root}" >> "${PLAN_FILE}"

  echo
  echo "================================================================"
  echo "[${combination_index}] ${subtype} | ${town} | ${weather}"
  echo "Episodes : ${start_episode}..${end_episode} (${episodes})"
  echo "Samples  : $((episodes * SAMPLES_PER_EPISODE))"
  echo "Output   : ${output_root}"
  echo "================================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  combo_report_root="${REPORT_ROOT}/combination_reports/${subtype}_${town}_${weather}"

  # On a resumed run, preserve incomplete attempts before reclaiming their
  # episode numbers. Completed episodes remain in place and are revalidated.
  for ((episode = start_episode; episode <= end_episode; episode++)); do
    suffix=$(printf "%03d" "${episode}")
    episode_dir="${output_root}/episode_${suffix}"
    if [[ -d "${episode_dir}" && ! -f "${episode_dir}/episode_manifest.json" ]]; then
      archive_root="${output_root}/failed_attempts/${RUN_ID}"
      mkdir -p "${archive_root}"
      echo "Preserving incomplete attempt: ${episode_dir}"
      mv "${episode_dir}" "${archive_root}/episode_${suffix}"
    fi
  done

  OUTPUT_ROOT="${output_root}" \
  START_EPISODE="${start_episode}" \
  EPISODES="${episodes}" \
  SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE}" \
  INTER_EPISODE_DELAY_SECONDS="${INTER_EPISODE_DELAY_SECONDS}" \
  RUN_TAG="emergency_12100_v1_0" \
  TOWN_OVERRIDE="${town}" \
  WEATHER_OVERRIDE="${weather}" \
  COLLECTOR_TIMEOUT="${COLLECTOR_TIMEOUT}" \
  MAP_LOAD_SETTLE_SECONDS="${MAP_LOAD_SETTLE_SECONDS}" \
  REPORT_ROOT_OVERRIDE="${combo_report_root}" \
    bash carla/self_collection/scripts/collect_resilient_batch.sh "${selector}"

  # The resilient child records validator outcomes for every requested episode.
  # Aggregate those outcomes instead of treating file existence as success.
  combo_summary="${combo_report_root}/batch_summary.tsv"
  for ((episode = start_episode; episode <= end_episode; episode++)); do
    suffix=$(printf "%03d" "${episode}")
    episode_dir="${output_root}/episode_${suffix}"
    episode_status="$(
      awk -F $'\t' -v target="${suffix}" \
        'NR > 1 && $1 == target { status=$3 } END { print status }' \
        "${combo_summary}" 2>/dev/null
    )"
    if [[ "${episode_status}" == "PASS" ]]; then
      pass_episodes=$((pass_episodes + 1))
      printf "%s\t%s\t%s\t%s\tPASS\t%s\t%s\n" \
        "${subtype}" "${town}" "${weather}" "${suffix}" \
        "${SAMPLES_PER_EPISODE}" "${episode_dir}" >> "${SUMMARY_FILE}"
    else
      fail_episodes=$((fail_episodes + 1))
      failure_reason="$(
        awk -F $'\t' -v target="${suffix}" \
          'NR > 1 && $1 == target { reason=$5 } END { print reason }' \
          "${combo_summary}" 2>/dev/null
      )"
      if [[ -z "${failure_reason}" ]]; then
        failure_reason="no PASS result in combination report"
      fi
      failure_reason=${failure_reason//$'\t'/ }
      printf "%s\t%s\t%s\t%s\tFAIL\t0\t%s\n" \
        "${subtype}" "${town}" "${weather}" "${suffix}" \
        "${episode_dir}" >> "${SUMMARY_FILE}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${subtype}" "${town}" "${weather}" "${suffix}" \
        "${failure_reason}" \
        "${episode_dir}" >> "${FAILURE_FILE}"
    fi
  done

  sleep "${INTER_COMBINATION_DELAY_SECONDS}"
done <<< "${PLAN_MATRIX}"

if (( selected_combinations == 0 )); then
  echo "No plan rows matched ONLY_TOWN='${ONLY_TOWN}' ONLY_SUBTYPE='${ONLY_SUBTYPE}'." >&2
  exit 2
fi

echo
echo "================================================================"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Emergency production dry run"
else
  echo "Emergency production batch finished"
fi
echo "Combinations     : ${selected_combinations}"
echo "Planned episodes : ${planned_episodes}"
echo "Planned samples  : ${planned_samples}"
if [[ "${DRY_RUN}" != "1" ]]; then
  echo "PASS episodes    : ${pass_episodes}"
  echo "FAIL episodes    : ${fail_episodes}"
  echo "Valid samples    : $((pass_episodes * SAMPLES_PER_EPISODE))"
  echo "Missing samples  : $((fail_episodes * SAMPLES_PER_EPISODE))"
fi
echo "Plan             : ${PLAN_FILE}"
echo "Summary          : ${SUMMARY_FILE}"
echo "Failures         : ${FAILURE_FILE}"
echo "================================================================"

# Episode-level failures are data-quality outcomes, not an interrupted batch.
# They are returned in failed_episodes.tsv and can be retried after inspection.
exit 0
