# CARLA 自采样场景分类（18 类）

## 1. 基础驾驶（B1–B7）

| 编号 | 场景 | 脚本参数 | 验收证据 |
|---|---|---|---|
| B1 | 30 km/h 巡航 | `basic` | 达到目标速度 |
| B2 | 40 km/h 巡航 | `basic_40` | 达到目标速度 |
| B3 | 加速至 60 km/h | `basic_60` | 在完整评价窗口达到目标速度 |
| B4 | 减速、停车、重新起步 | `basic_stop_restart` | 车速低于 0.25 m/s 后重新超过 2 m/s |
| B5 | 左转 | `basic_turn_left` | 航向变化不少于 35° |
| B6 | 右转 | `basic_turn_right` | 航向变化不少于 30° |
| B7 | 主动变道 | `basic_lane_change` | 连续 5 帧处于目标相邻车道 |

## 2. 复杂交互（C1–C5）

| 编号 | 场景 | 脚本参数 | 验收证据 |
|---|---|---|---|
| C1 | 混合交通 | `complex` | 多类交通参与者及完整多模态样本 |
| C2 | 常规行人过街 | `complex_pedestrian` | 行人完成横穿，自车正常让行且无碰撞 |
| C3 | 自行车/摩托车流 | `complex_two_wheeler` | 至少 2 辆摩托车、2 辆自行车及 6 辆普通车生成成功 |
| C4 | 路口多方向交通 | `complex_intersection` | 至少 10 辆背景车且自车完成路口转向 |
| C5 | 静止障碍物避让 | `complex_static_obstacle` | 障碍物真实生成，自车进入相邻车道且无碰撞 |

## 3. 极限应急（E1–E4，含 E2 三级强度）

| 编号 | 场景 | 脚本参数 | 验收证据 |
|---|---|---|---|
| E1 | 车辆突然切入 | `extreme_e1` | 切入车进入自车道，自车响应且无碰撞 |
| E2-safe | 行人横穿 20–25 m | `extreme_e2` | 事件完成、制动响应、无碰撞 |
| E2-hard | 行人横穿 14–20 m | `extreme_e2_hard` | 触发距离、响应时间和制动强度满足 hard 阈值 |
| E2-critical | 行人横穿 10–14 m | `extreme_e2_critical` | 触发距离、响应时间和最大制动满足 critical 阈值 |
| E3 | 施工收窄与并道 | `extreme_e3` | 锥桶生成、自车完成并道、无碰撞 |
| E4 | 前车突然急刹 | `extreme_e4` | 前车真实减速、自车及时响应、无碰撞 |

## 运行方式

在 WSL 的 OpenDriveVLA 仓库根目录运行单类小试：

```bash
OUTPUT_ROOT="/path/to/carla_self_collection/pilot_B4" \
EPISODES=1 SAMPLES_PER_EPISODE=10 \
bash carla/self_collection/scripts/collect_small_scale.sh basic_stop_restart
```

只在 CARLA 窗口演示、不写文件：

```bash
python carla/self_collection/collectors/multimodal_collect.py \
  --config carla/self_collection/scenarios/basic_stop_restart.pilot.json \
  --visual-only --visual-duration 15 \
  --vehicles 8 --walkers 2 --motorcycles 0 --bicycles 0 \
  --nearby-radius 60 --seed 6201
```

`collect_small_scale.sh taxonomy_all` 会按顺序运行全部 18 个小类。脚本默认检查
`episode_manifest.json`：完整且已存在的 episode 会重新验收后跳过；非空但不完整的目录会停止运行，避免覆盖数据。

## 正式扩量前的门槛

新增 B4–B7、C2–C5 和 E2-hard 必须先各运行 1 个 episode，并确认终端显示 `PASS: 10/10`。
转向类还需要在 CARLA 窗口人工确认转向方向正确。全部通过后才能开始 33k 帧扩量。
