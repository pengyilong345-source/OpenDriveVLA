# CARLA 自采样语义事件控制器

## 设计依据

本控制器参考本地 Bench2Drive ScenarioRunner 的设计原则：

- 背景交通与场景 Actor 分开生成。
- 场景 Actor 使用独立的 `scenario` 角色。
- 事件通过距离或到达时间触发，不依赖随机交通碰巧制造危险。
- 使用 ego 响应、最小距离和碰撞状态进行验收。

当前参考的 Bench2Drive 实现包括 `FollowLeadingVehicle`、`HighwayCutIn`、
`Accident` 和 `StaticCutIn`。自采样模块不直接依赖 ScenarioRunner，而是在轻量
Collector 中实现相同的“场景 Actor + 触发器 + 验收指标”结构。

## 已实现事件

### `lead_vehicle_hard_brake`

1. 在 ego 当前车道前方约 25 m 生成一辆四轮场景车。
2. 场景车使用 Traffic Manager 低速行驶且禁止自动变道。
3. episode 开始 2 s 后进入触发窗口。
4. 场景车距离 ego 不超过 16 m 时立即退出自动驾驶并全力制动。
5. 如果 3 s 后仍未进入 16 m，但位于 ego 前方 35 m 内，则强制触发。
6. 持续制动 2.5 s，然后保持驻车。
7. 根据 ego 制动、纵向减速度和碰撞状态判断事件是否成功。

配置位于：

```text
scenarios/extreme_emergency.pilot.json
```

### `adjacent_vehicle_cut_in`

1. 在 ego 左侧或右侧相邻车道前方约 10 m 生成一辆四轮场景车。
2. episode 预热完成后进入触发窗口，并通过 Traffic Manager 强制向 ego 车道变道。
3. 根据 ego 坐标系中的横向距离判断场景车是否真正进入 ego 车道，不依赖固定道路或车道 ID。
4. 根据 ego 制动、纵向减速度、最小车距和碰撞状态判断事件是否成功。
5. 若车辆未在规定时间内进入 ego 车道，或发生碰撞，则 episode 验收失败。

配置位于：

```text
scenarios/extreme_cut_in.pilot.json
```

## 验收条件

极限应急 episode 必须同时满足：

- `event.required = true`
- `event.triggered = true`
- `event.state = completed`
- `event.success = true`
- `event.event_actor_id` 存在
- ego 最大制动不小于配置阈值，或纵向减速度达到配置阈值
- `event.collision = false`

任一条件不满足时，该 episode 的 `sample_valid` 为 `false`，验证脚本返回失败，
不能进入正式训练数据集。

## 输出指标

每条自采样标注和 `episode_manifest.json` 都记录：

- 事件类型、状态、是否触发和是否成功
- 场景 Actor ID、触发帧和触发时间
- 当前距离和最小距离
- 场景 Actor 速度
- ego 最大制动和最小纵向加速度
- 是否碰撞及碰撞 Actor ID
- 失败原因

## 当前边界

当前已实现并通过 CARLA 实机小规模验收的事件为前车急刹和相邻车切入。以下
Bench2Drive 类事件仍需逐项增加控制器，并分别完成
CARLA 实机验收：

- 事故占道（Accident / AccidentTwoWays）
- 施工障碍（ConstructionObstacle）
- 行人或自行车横穿
- 紧急车辆让行

在这些控制器完成前，不得仅凭随机 Actor 数量为样本标注对应语义事件。
