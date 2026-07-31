# 极限应急场景与验收规范

本目录按比赛方案中的极限应急工况建立四类可重复的 CARLA 语义事件。四类场景统一使用
`rainy_night` 天气（夜间、强降雨、积水反光和雨雾），但天气只作为环境条件，不能代替动态事件本身。

## E1 突发车辆加塞

- 配置：`scenarios/extreme_e1_cut_in.pilot.json`
- 事件类型：`adjacent_vehicle_cut_in`
- 生成：在同向相邻车道、自车前方生成场景车辆并强制向自车车道切入。
- 事件证据：场景车辆进入自车所在 road/lane，并在自车前方横向容差内持续达到规定帧数。
- 响应证据：事件触发后，自车制动或纵向减速度相对触发前基线产生足够增量。

## E2 临时横穿行人

- 配置：`scenarios/extreme_e2_pedestrian_crossing.pilot.json`
- 事件类型：`pedestrian_crossing`
- 生成：在自车前方道路一侧生成专用行人，触发后横穿当前车道。
- 事件证据：行人越过车道中心并到达另一侧方向，持续达到规定帧数。
- 响应证据：自车产生紧急制动或明显纵向减速度，且无自车/行人碰撞。

## E3 施工锥桶与车道收窄

- 配置：`scenarios/extreme_e3_construction_merge.pilot.json`
- 事件类型：`construction_lane_narrowing`
- 生成：在当前车道前方横向布置施工锥桶，并只选择存在同向左车道的出生点。
- 事件证据：自车进入目标左车道并持续达到规定帧数。
- 响应证据：自车先减速并完成左并道，全程无碰撞。
- 当前 Pilot 配置设有 `assist_ego_lane_change=true`，用于生成控制器示范数据；接入 VLA 闭环评测时必须改为
  `false`，由模型独立输出并道动作。

## E4 前方突发危险（前车急刹实例）

- 配置：`scenarios/extreme_e4_lead_hard_brake.pilot.json`
- 事件类型：`lead_vehicle_hard_brake`
- 生成：前车正常行驶进入触发距离后退出自动驾驶并全力制动。
- 事件证据：记录前车触发前速度、事件中最低速度和速度下降量；速度下降不足时不得通过。
- 响应证据：自车产生相对触发前基线的制动/减速度增量并保持无碰撞。

## 公共落盘指标

每个样本和 `episode_manifest.json` 中的 `event` 至少提供：

- 事件状态、触发帧和触发时间；
- 当前距离、最小距离和最小 TTC；
- 自车触发前基线、最大制动、最小纵向加速度；
- 物理响应时间及其是否处于配置窗口；
- 事件 Actor 初始/最低/最终速度和速度下降量（适用时）；
- 起始车道、目标车道、驻留帧数和动作证据；
- 自车碰撞和事件 Actor 碰撞；
- 是否使用控制器辅助动作。

这里的 `response_latency_seconds` 是 CARLA 中“事件触发到车辆物理响应”的时间，不等同于比赛要求的
VLA 推理延时。比赛的 `<=120 ms` 仍需在模型接口处独立记录“指令/传感器输入到动作输出”的耗时。

## Pilot 运行顺序

在 Windows 已启动一个 CARLA Server、WSL 已激活 CARLA 0.9.15 Python 环境后，从仓库根目录运行：

只在 CARLA 窗口中演示 E2、不挂载采样传感器且不写数据：

```bash
python carla/self_collection/collectors/multimodal_collect.py \
  --config carla/self_collection/scenarios/extreme_e2_pedestrian_crossing.pilot.json \
  --visual-only --visual-duration 15 \
  --vehicles 8 --walkers 1 --nearby-radius 70 --seed 7201
```

视觉演示结束后会自动删除本次 ego、背景交通、行人和场景 Actor，但保留 CARLA Server。

```bash
EPISODES=1 SAMPLES_PER_EPISODE=5 RUN_TAG=extreme_e1 \
  bash carla/self_collection/scripts/collect_small_scale.sh extreme_e1
```

依次将最后的参数改为 `extreme_e2`、`extreme_e3`、`extreme_e4`。四类单独通过后再运行：

```bash
EPISODES=3 SAMPLES_PER_EPISODE=5 RUN_TAG=extreme_all \
  bash carla/self_collection/scripts/collect_small_scale.sh extreme_all
```

在每类至少连续三个 episode 通过实机验收前，这些配置保持 `.pilot.json` 命名，不应宣称为正式训练集。
