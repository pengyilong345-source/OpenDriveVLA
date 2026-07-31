#!/usr/bin/env bash
set -euo pipefail

# Sequentially replay the currently usable taxonomy routes in CARLA without
# attaching data sensors or writing sample files. Run from the repository root.

START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-17}"
VISUAL_DURATION="${VISUAL_DURATION:-15}"
INTER_ROUTE_DELAY_SECONDS="${INTER_ROUTE_DELAY_SECONDS:-2}"

if (( START_INDEX < 1 || END_INDEX > 17 || START_INDEX > END_INDEX )); then
  echo "START_INDEX/END_INDEX must satisfy 1 <= START_INDEX <= END_INDEX <= 17" >&2
  exit 2
fi

review_route() {
  local index="$1"
  local code="$2"
  local description="$3"
  local config="$4"
  local vehicles="$5"
  local walkers="$6"
  local motorcycles="$7"
  local bicycles="$8"
  local radius="$9"
  local seed="${10}"

  if (( index < START_INDEX || index > END_INDEX )); then
    return
  fi

  echo
  echo "================================================================"
  echo "[$index/17] $code - $description"
  echo "配置：$config"
  echo "只在 CARLA 中运行最多 ${VISUAL_DURATION}s，不写采样文件"
  echo "================================================================"

  python carla/self_collection/collectors/multimodal_collect.py \
    --config "$config" \
    --visual-only \
    --visual-duration "$VISUAL_DURATION" \
    --vehicles "$vehicles" \
    --walkers "$walkers" \
    --motorcycles "$motorcycles" \
    --bicycles "$bicycles" \
    --nearby-radius "$radius" \
    --seed "$seed"

  sleep "$INTER_ROUTE_DELAY_SECONDS"
}

review_route 1  B1 "30 km/h 巡航" \
  carla/self_collection/scenarios/basic_control.pilot.json 8 3 0 0 60 1101
review_route 2  B2 "40 km/h 巡航" \
  carla/self_collection/scenarios/basic_cruise_40.pilot.json 8 3 0 0 60 5101
review_route 3  B3 "加速至 60 km/h" \
  carla/self_collection/scenarios/basic_accelerate_60.pilot.json 8 0 0 0 60 6101
review_route 4  B4 "减速、停车、重新起步" \
  carla/self_collection/scenarios/basic_stop_restart.pilot.json 8 2 0 0 60 6201
review_route 5  B5 "左转" \
  carla/self_collection/scenarios/basic_turn_left.pilot.json 8 2 0 0 70 6301
review_route 6  B6 "右转" \
  carla/self_collection/scenarios/basic_turn_right.pilot.json 8 2 0 0 70 6401
review_route 7  B7 "主动变道" \
  carla/self_collection/scenarios/basic_lane_change.pilot.json 8 2 0 0 70 6502

review_route 8  C1 "城市拥堵混合交通流" \
  carla/self_collection/scenarios/complex_obstacle_avoidance.pilot.json 24 12 3 3 55 2201
review_route 9  C2 "常规行人过街" \
  carla/self_collection/scenarios/complex_pedestrian_crossing.pilot.json 12 4 0 0 80 2301
review_route 10 C3 "自行车和摩托车流" \
  carla/self_collection/scenarios/complex_two_wheeler_flow.pilot.json 12 2 4 4 90 2401
review_route 11 C4 "路口多方向交通" \
  carla/self_collection/scenarios/complex_intersection_traffic.pilot.json 18 6 2 2 90 2503
review_route 12 C5 "静态障碍物避让" \
  carla/self_collection/scenarios/complex_static_obstacle_avoidance.pilot.json 12 3 0 0 80 2602

review_route 13 E1 "车辆突然切入" \
  carla/self_collection/scenarios/extreme_e1_cut_in.pilot.json 10 2 0 0 70 7101
review_route 14 E2-safe "行人安全距离横穿" \
  carla/self_collection/scenarios/extreme_e2_pedestrian_crossing.pilot.json 8 1 0 0 70 7201
review_route 15 E2-hard "行人短距离横穿" \
  carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_hard.pilot.json 8 1 0 0 70 7231
review_route 16 E2-critical "行人临界距离横穿" \
  carla/self_collection/scenarios/extreme_e2_pedestrian_crossing_critical.pilot.json 8 1 0 0 70 7251
review_route 17 E4 "前车突然急刹" \
  carla/self_collection/scenarios/extreme_e4_lead_hard_brake.pilot.json 10 2 0 0 70 7401

echo
echo "人工巡检范围已运行完毕；整个过程未写入采样文件。"
echo "E3 施工收窄尚未形成正式有效样本，本轮未纳入。"
