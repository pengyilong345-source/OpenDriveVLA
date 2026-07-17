#!/usr/bin/env bash
set -euo pipefail

# Run the official evaluator against a CARLA server already running on Windows.
# The upstream Bench2Drive checkout remains untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:?Set BENCH2DRIVE_ROOT to the external Bench2Drive checkout}"
CARLA_PYTHONAPI="${CARLA_PYTHONAPI:?Set CARLA_PYTHONAPI to CARLA 0.9.15/PythonAPI/carla}"
CARLA_HOST="${CARLA_HOST:-$(ip route | awk '/default/ {print $3; exit}')}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
ROUTE_SUBSET="${ROUTE_SUBSET:-2091}"
DEBUG_CHALLENGE="${DEBUG_CHALLENGE:-1}"
ROUTES="${ROUTES:-${BENCH2DRIVE_ROOT}/leaderboard/data/drivetransformer_bench2drive_dev10.xml}"
AGENT="${AGENT:-${BENCH2DRIVE_ROOT}/leaderboard/leaderboard/autoagents/npc_agent.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/carla/output/bench2drive_preflight}"
EXTERNAL_SERVER_STUB="${EXTERNAL_SERVER_STUB:-${SCRIPT_DIR}/external_server_stub}"

if [[ ! -d "${BENCH2DRIVE_ROOT}" ]]; then
  echo "Bench2Drive root not found: ${BENCH2DRIVE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CARLA_PYTHONAPI}" ]]; then
  echo "CARLA PythonAPI not found: ${CARLA_PYTHONAPI}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${CARLA_PYTHONAPI}:${BENCH2DRIVE_ROOT}:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/scenario_runner:${PYTHONPATH:-}"
export SCENARIO_RUNNER_ROOT="${BENCH2DRIVE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${BENCH2DRIVE_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME="SENSORS"
export CARLA_ROOT="${EXTERNAL_SERVER_STUB}"

echo "Bench2Drive root: ${BENCH2DRIVE_ROOT}"
echo "CARLA PythonAPI:   ${CARLA_PYTHONAPI}"
echo "CARLA server:     ${CARLA_HOST}:${CARLA_PORT}"
echo "Server stub:      ${EXTERNAL_SERVER_STUB}"
echo "Routes:           ${ROUTES}"
echo "Route subset:     ${ROUTE_SUBSET}"
echo "Output:           ${OUTPUT_ROOT}"

cd "${BENCH2DRIVE_ROOT}"

python "${BENCH2DRIVE_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py" \
  --host="${CARLA_HOST}" \
  --port="${CARLA_PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --traffic-manager-seed=0 \
  --routes="${ROUTES}" \
  --routes-subset="${ROUTE_SUBSET}" \
  --repetitions=1 \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${OUTPUT_ROOT}/results.json" \
  --debug-checkpoint="${OUTPUT_ROOT}/live_results.txt" \
  --agent="${AGENT}" \
  --agent-config="" \
  --debug="${DEBUG_CHALLENGE}" \
  --timeout=120
