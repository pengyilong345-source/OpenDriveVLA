#!/usr/bin/env bash
set -uo pipefail

# Qualify emergency subtypes on multiple CARLA maps before production sampling.
# Each combination gets an isolated output directory. A failed combination is
# recorded and the matrix continues; no production_33k_v1_0 data is touched.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"

QUALIFICATION_ROOT="${QUALIFICATION_ROOT:-${WORKSPACE_ROOT}/data/carla_self_collection/emergency_map_qualification_v1}"
TOWNS="${TOWNS:-Town01 Town03 Town05 Town10HD_Opt}"
WEATHERS="${WEATHERS:-clear_day}"
SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-10}"
INTER_COMBINATION_DELAY_SECONDS="${INTER_COMBINATION_DELAY_SECONDS:-2}"
RUN_ID="${RUN_ID:-map_clear_v1}"
EPISODE_NUMBER="${EPISODE_NUMBER:-1}"
RETRY_FAILURES_FILE="${RETRY_FAILURES_FILE:-}"

RUN_ROOT="${QUALIFICATION_ROOT}/runs/${RUN_ID}"
REPORT_ROOT="${RUN_ROOT}/reports"
SUMMARY_FILE="${REPORT_ROOT}/qualification_summary.tsv"
WHITELIST_FILE="${REPORT_ROOT}/qualified_combinations.tsv"
FAILURE_FILE="${REPORT_ROOT}/failed_combinations.tsv"

mkdir -p "${REPORT_ROOT}"
printf "subtype\tselector\ttown\tweather\tstatus\texit_code\toutput\n" > "${SUMMARY_FILE}"
printf "subtype\tselector\ttown\tweather\toutput\n" > "${WHITELIST_FILE}"
printf "subtype\tselector\ttown\tweather\texit_code\treason\toutput\n" > "${FAILURE_FILE}"

cd "${REPO_ROOT}"

attempted=0
passed=0
failed=0

run_combination() {
  local subtype="$1"
  local selector="$2"
  local town="$3"
  local weather="$4"
  local output_root="${RUN_ROOT}/${subtype}/${town}_${weather}"
  local log_file="${REPORT_ROOT}/${subtype}_${town}_${weather}.log"

  attempted=$((attempted + 1))
  echo
  echo "============================================================"
  echo "[${attempted}] ${subtype} | ${town} | ${weather}"
  echo "Output: ${output_root}"
  echo "============================================================"

  OUTPUT_ROOT="${output_root}" \
  START_EPISODE="${EPISODE_NUMBER}" \
  EPISODES=1 \
  SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE}" \
  INTER_EPISODE_DELAY_SECONDS=0 \
  SKIP_EXISTING=1 \
  RUN_TAG="emergency_map_qualification" \
  TOWN_OVERRIDE="${town}" \
  WEATHER_OVERRIDE="${weather}" \
  COLLECTOR_TIMEOUT=90 \
  MAP_LOAD_SETTLE_SECONDS=10 \
    bash carla/self_collection/scripts/collect_small_scale.sh "${selector}" \
      2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}

  if (( status == 0 )); then
    passed=$((passed + 1))
    printf "%s\t%s\t%s\t%s\tPASS\t0\t%s\n" \
      "${subtype}" "${selector}" "${town}" "${weather}" "${output_root}" >> "${SUMMARY_FILE}"
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${subtype}" "${selector}" "${town}" "${weather}" "${output_root}" >> "${WHITELIST_FILE}"
    echo "QUALIFIED: ${subtype} on ${town} / ${weather}"
  else
    failed=$((failed + 1))
    local reason
    reason=$(
      grep -E "RuntimeError:|ValueError:|FileNotFoundError:|FAIL:|Fatal error|TimeoutError:" \
        "${log_file}" |
        tail -n 1
    )
    if [[ -z "${reason}" ]]; then
      reason="collector exited with code ${status}"
    fi
    reason=${reason//$'\t'/ }
    printf "%s\t%s\t%s\t%s\tFAIL\t%s\t%s\n" \
      "${subtype}" "${selector}" "${town}" "${weather}" "${status}" "${output_root}" >> "${SUMMARY_FILE}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${subtype}" "${selector}" "${town}" "${weather}" "${status}" "${reason}" "${output_root}" >> "${FAILURE_FILE}"
    echo "REJECTED: ${subtype} on ${town} / ${weather} (continuing)"
  fi

  sleep "${INTER_COMBINATION_DELAY_SECONDS}"
}

if [[ -n "${RETRY_FAILURES_FILE}" ]]; then
  if [[ ! -f "${RETRY_FAILURES_FILE}" ]]; then
    echo "Failure list not found: ${RETRY_FAILURES_FILE}" >&2
    exit 2
  fi
  echo "Retrying only combinations listed in: ${RETRY_FAILURES_FILE}"
  while IFS=$'\t' read -r subtype selector town weather _exit_code _reason _output; do
    if [[ "${subtype}" == "subtype" || -z "${subtype}" ]]; then
      continue
    fi
    run_combination "${subtype}" "${selector}" "${town}" "${weather}"
  done < "${RETRY_FAILURES_FILE}"
else
  for weather in ${WEATHERS}; do
    for town in ${TOWNS}; do
      run_combination "E1_vehicle_cut_in" "extreme_e1" "${town}" "${weather}"
      run_combination "E2_safe_pedestrian" "extreme_e2" "${town}" "${weather}"
      run_combination "E2_hard_pedestrian" "extreme_e2_hard" "${town}" "${weather}"
      run_combination "E2_critical_pedestrian" "extreme_e2_critical" "${town}" "${weather}"
      run_combination "E3_construction_merge" "extreme_e3" "${town}" "${weather}"
      run_combination "E4_lead_hard_brake" "extreme_e4" "${town}" "${weather}"
    done
  done
fi

echo
echo "============================================================"
echo "Emergency map qualification finished"
echo "Attempted : ${attempted}"
echo "PASS      : ${passed}"
echo "FAIL      : ${failed}"
echo "Summary   : ${SUMMARY_FILE}"
echo "Whitelist : ${WHITELIST_FILE}"
echo "Failures  : ${FAILURE_FILE}"
echo "============================================================"

exit 0
