#!/usr/bin/env bash
set -euo pipefail

# Run one selected Bench2Drive route for visual inspection without recording data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:?Set BENCH2DRIVE_ROOT to the external Bench2Drive checkout}"
ROUTE_ID="${1:?Usage: bash carla/bench2drive/run_demo_route.sh ROUTE_ID}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

export ROUTES="${PROJECT_ROOT}/carla/scenarios/bench2drive_finetune_candidates_v1.xml"
export ROUTE_SUBSET="${ROUTE_ID}"
export AGENT="${BENCH2DRIVE_ROOT}/leaderboard/leaderboard/autoagents/npc_agent.py"
export OUTPUT_ROOT="${PROJECT_ROOT}/carla/output/bench2drive_demo/route_${ROUTE_ID}_${RUN_TAG}"
export DEBUG_CHALLENGE=0

echo "Demo only: route ${ROUTE_ID}; sensor dataset recording is disabled."
echo "Run tag: ${RUN_TAG}"
exec bash "${SCRIPT_DIR}/run_external_evaluator.sh"
