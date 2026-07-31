# CARLA Self-Collection

本目录存放 CARLA 0.9.15 自采样代码和配置，与 `carla/bench2drive/` 数据管线分开维护，
两者共用 `carla/schema/` 中的 sample-v1.1 格式。

## 目录

```text
self_collection/
├── collectors/  Python 采集和样本验证程序
├── configs/     后续拆分的交通、传感器和运行参数
├── scenarios/   三类比赛场景配置
├── scripts/     批量试采脚本
├── tests/       不依赖 CARLA Server 的事件状态机测试
├── EVENT_CONTROLLER.md
└── TRAFFIC_AND_ACTOR_CONFIG.md
```

## 语义事件

极限应急场景已接入配置驱动的 `lead_vehicle_hard_brake` 和
`adjacent_vehicle_cut_in` 事件。Collector 会生成专用场景车、主动触发急刹或切入，
并根据 ego 响应、事件是否完成和碰撞状态决定该 episode 是否有效。实现和验收规则见
[`EVENT_CONTROLLER.md`](EVENT_CONTROLLER.md)。

## 基础速度场景

基础操控现支持 Traffic Manager 的真实目标速度控制，不再只把目标速度写进标注：

- `basic_control.pilot.json`：30 km/h 安全巡航
- `basic_cruise_40.pilot.json`：40 km/h 保持车道
- `basic_accelerate_60.pilot.json`：按比赛方案示例提速至 60 km/h

每个速度 episode 会在 `episode_manifest.json` 中记录目标速度、样本速度范围、均值和
验收结果。车辆未达到目标速度容差时，样本会被标记为无效。

## 第一轮试采

在 OpenDriveVLA 仓库根目录、可导入 CARLA 0.9.15 Python Client 的环境中运行：

```bash
EPISODES=3 SAMPLES_PER_EPISODE=5 \
  bash carla/self_collection/scripts/collect_small_scale.sh basic
```

依次将最后一个参数替换为 `complex`、`emergency_brake` 和 `emergency_cutin`。
各类场景都验证通过后，
再使用 `all` 或提高 episode、样本数量。

如需将数据写入仓库外，显式设置 `OUTPUT_ROOT`：

```bash
OUTPUT_ROOT=/path/to/data/carla_self_collection/basic_control \
EPISODES=3 SAMPLES_PER_EPISODE=5 \
  bash carla/self_collection/scripts/collect_small_scale.sh basic
```

交通密度、随机种子和每个 episode 的记录要求见
[`TRAFFIC_AND_ACTOR_CONFIG.md`](TRAFFIC_AND_ACTOR_CONFIG.md)。

从环境检查到正式扩量的逐项执行流程见
[`SAMPLING_CHECKLIST.md`](SAMPLING_CHECKLIST.md)。

## 注意事项

- Windows 运行 CARLA Server，默认 RPC 端口为 `2000`。
- Collector 默认动态获取 WSL 看到的 Windows 网关地址。
- 输出 episode 目录必须不存在或为空，脚本不会覆盖已有采样。
- 中断或异常退出后，应确认 ego、NPC、行人控制器和传感器均已销毁。
