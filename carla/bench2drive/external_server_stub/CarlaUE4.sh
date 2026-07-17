#!/usr/bin/env bash
set -euo pipefail

# The real CARLA server is managed on Windows. Bench2Drive expects to own a
# CarlaUE4.sh process, so keep this lightweight placeholder alive until the
# evaluator terminates its process group.
exec sleep infinity
