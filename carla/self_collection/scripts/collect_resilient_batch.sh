#!/usr/bin/env bash
set -uo pipefail

# Run a large CARLA collection batch without aborting on a single failed episode.
# Completed episodes are revalidated and skipped by collect_small_scale.sh.
# Failed episodes remain in place for inspection and are listed in a timestamped
# report. This script intentionally exits 0 after the batch so that failures can
# be reviewed and retried separately.

SCENARIO="${1:?Usage: collect_resilient_batch.sh <scenario>}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the category output directory}"
START_EPISODE="${START_EPISODE:-1}"
EPISODES="${EPISODES:-1}"
SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-10}"
INTER_EPISODE_DELAY_SECONDS="${INTER_EPISODE_DELAY_SECONDS:-2}"
RUN_TAG="${RUN_TAG:-production}"
REPORT_ROOT_OVERRIDE="${REPORT_ROOT_OVERRIDE:-}"

if (( EPISODES < 1 )); then
  echo "EPISODES must be at least 1" >&2
  exit 2
fi

end_episode=$((START_EPISODE + EPISODES - 1))
run_id=$(date +"%Y%m%d_%H%M%S")
if [[ -n "${REPORT_ROOT_OVERRIDE}" ]]; then
  report_root="${REPORT_ROOT_OVERRIDE}"
else
  report_root="${OUTPUT_ROOT}/reports/${run_id}"
fi
summary_file="${report_root}/batch_summary.tsv"
failed_file="${report_root}/failed_episodes.tsv"

mkdir -p "${report_root}"
printf "episode\tseed\tstatus\texit_code\treason\toutput\n" > "${summary_file}"
printf "episode\tseed\texit_code\treason\toutput\n" > "${failed_file}"

success_count=0
failure_count=0

echo "============================================================"
echo "Resilient CARLA batch"
echo "Scenario : ${SCENARIO}"
echo "Episodes : ${START_EPISODE}..${end_episode} (${EPISODES} attempts)"
echo "Samples  : ${SAMPLES_PER_EPISODE} per episode"
echo "Output   : ${OUTPUT_ROOT}"
echo "Report   : ${report_root}"
echo "============================================================"

for ((episode = START_EPISODE; episode <= end_episode; episode++)); do
  suffix=$(printf "%03d" "${episode}")
  episode_output="${OUTPUT_ROOT}/episode_${suffix}"
  episode_log="${report_root}/episode_${suffix}.log"

  echo
  echo "========== Episode ${suffix}/${end_episode} =========="

  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  START_EPISODE="${episode}" \
  EPISODES=1 \
  SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE}" \
  INTER_EPISODE_DELAY_SECONDS=0 \
  RUN_TAG="${RUN_TAG}" \
  bash carla/self_collection/scripts/collect_small_scale.sh "${SCENARIO}" \
    2>&1 | tee "${episode_log}"
  status=${PIPESTATUS[0]}

  seed=$(
    grep -E -- "--> .* episode ${suffix}, seed=" "${episode_log}" |
      tail -n 1 |
      sed -E 's/.*seed=([0-9]+).*/\1/'
  )
  if [[ -z "${seed}" ]]; then
    seed="unknown"
  fi

  if (( status == 0 )); then
    success_count=$((success_count + 1))
    printf "%s\t%s\tPASS\t0\t-\t%s\n" \
      "${suffix}" "${seed}" "${episode_output}" >> "${summary_file}"
    echo "Episode ${suffix}: PASS"
  else
    failure_count=$((failure_count + 1))
    reason=$(
      grep -E "RuntimeError:|ValueError:|FileNotFoundError:|FAIL:|Fatal error|TimeoutError:" \
        "${episode_log}" |
        tail -n 1
    )
    if [[ -z "${reason}" ]]; then
      reason="collector exited with code ${status}"
    fi
    reason=${reason//$'\t'/ }
    printf "%s\t%s\tFAIL\t%s\t%s\t%s\n" \
      "${suffix}" "${seed}" "${status}" "${reason}" "${episode_output}" >> "${summary_file}"
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${suffix}" "${seed}" "${status}" "${reason}" "${episode_output}" >> "${failed_file}"
    echo "Episode ${suffix}: FAIL (recorded; continuing)"
  fi

  if (( episode < end_episode )); then
    sleep "${INTER_EPISODE_DELAY_SECONDS}"
  fi
done

echo
echo "============================================================"
echo "Batch finished"
echo "Attempted : ${EPISODES}"
echo "PASS      : ${success_count}"
echo "FAIL      : ${failure_count}"
echo "Summary   : ${summary_file}"
echo "Failures  : ${failed_file}"
echo "============================================================"

# A non-zero exit here would make an overnight batch look interrupted even
# though every requested episode was attempted. Failure details are preserved
# in failed_episodes.tsv for the dedicated retry pass.
exit 0
