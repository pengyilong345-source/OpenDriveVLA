# CARLA 自采样 Command 对齐规范（v1）

## 1. 目的

本文件用于统一 CARLA 自采样数据中的：

- 场景小类（taxonomy subtype）
- 中文自然语言驾驶指令（`command_text`）
- 指令类别（`command_type`）
- 驾驶意图（`intent_label`）
- 目标车速与目标车道
- CARLA 中实际发生的驾驶行为

对齐的基本要求是：**文本说什么，场景中就必须发生什么，样本标注也必须记录什么。**

比赛方案依据：

- 第 3 页：数据集应包含单步、组合、应急指令，并构建“语音 + 视觉 + 车辆状态”多模态数据集。
- 第 5 页：模型包括语音解析、视觉理解、语义对齐、动作生成四个模块。
- 第 5 页：需要说明语音、视觉、车辆状态的对齐方法，以及数据构成、预处理和标注规范。

## 2. 指令类别定义

| `command_type` | 含义 | 示例 |
|---|---|---|
| `single` | 一个明确的基础驾驶动作 | 在前方路口左转 |
| `compound` | 包含观察、速度控制、车道控制等两个及以上连续或并行意图 | 确认安全后变更至相邻车道 |
| `emergency` | 由突发危险触发、要求立即响应的指令 | 前车突然急刹，立即制动并保持安全距离 |

`command_type` 描述的是指令语义，不等同于数据集的三大场景类别。基础场景也可以包含组合指令。

## 3. 18 个已验收小类的标准 Command

### 3.1 基础操作（B，7 类）

| ID | 场景 | 标准中文指令 | 类型 | 标准意图标签 | 应与 CARLA 行为对齐 |
|---|---|---|---|---|---|
| B1 | 30 km/h 巡航 | 沿当前车道，以30公里每小时安全行驶 | `compound` | `lane_follow`, `maintain_speed` | 保持当前车道，速度稳定在约 8.33 m/s |
| B2 | 40 km/h 巡航 | 保持当前车道，以40公里每小时安全行驶 | `compound` | `lane_follow`, `maintain_speed` | 保持当前车道，速度稳定在约 11.11 m/s |
| B3 | 加速至 60 km/h | 保持当前车道，平稳提速至60公里每小时 | `compound` | `lane_follow`, `accelerate`, `target_speed` | 沿当前车道加速，最终接近 16.67 m/s |
| B4 | 停车后重新起步 | 平稳减速停车，短暂停留后重新起步 | `compound` | `decelerate`, `stop`, `restart` | 完成减速、静止停留和重新起步三个阶段 |
| B5 | 左转 | 在前方路口左转 | `single` | `turn_left` | 在指定目标路口左转，不能在后续路口继续错误转向 |
| B6 | 右转 | 在前方路口右转 | `single` | `turn_right` | 在指定目标路口右转 |
| B7 | 安全变道 | 观察相邻车道，确认安全后变更至相邻车道 | `compound` | `observe_adjacent_lane`, `check_gap`, `lane_change` | 检查目标车道空隙并完成变道；因正常红灯停车不能误判为变道失败 |

### 3.2 复杂交互与避障（C，5 类）

| ID | 场景 | 标准中文指令 | 类型 | 标准意图标签 | 应与 CARLA 行为对齐 |
|---|---|---|---|---|---|
| C1 | 城市拥堵混合交通 | 前方交通拥堵，注意周围车辆、非机动车和行人，低速跟车并保持安全距离 | `compound` | `observe_mixed_traffic`, `follow_congested_traffic`, `maintain_safe_distance`, `keep_lane` | ego 前方必须存在同车道队列，并呈现低速、停车、再起步的拥堵波动 |
| C2 | 常规行人过街 | 注意前方行人，减速并停车让行 | `compound` | `observe_pedestrian`, `decelerate`, `yield_pedestrian` | 行人真实进入 ego 行驶路径；ego 提前减速并无碰撞让行 |
| C3 | 自行车和摩托车流 | 注意前方及侧前方的自行车和摩托车，低速行驶并保持安全间距 | `compound` | `observe_two_wheelers`, `low_speed_following`, `maintain_safe_distance`, `keep_lane` | 两轮车分布在当前车道前方和侧前方，不能全部排成与 ego 无关的一列 |
| C4 | 路口多方向交通 | 观察路口多方向交通，确认安全后左转 | `compound` | `observe_cross_traffic`, `yield`, `turn_left` | 路口至少存在多个方向的相关车辆；ego 等待安全窗口后无碰撞左转 |
| C5 | 静止障碍物避让 | 前方有静止障碍物，减速并安全绕行 | `compound` | `observe_static_obstacle`, `decelerate`, `avoid_static_obstacle`, `lane_change` | 障碍物位于 ego 路径上；ego 减速、绕行并在通过后继续稳定行驶 |

### 3.3 极限应急（E，6 类）

| ID | 场景 | 标准中文指令 | 类型 | 标准意图标签 | 应与 CARLA 行为对齐 |
|---|---|---|---|---|---|
| E1 | 突发车辆加塞 | 邻车突然切入，立即减速避让并保持安全距离 | `emergency` | `emergency_decelerate`, `avoid_vehicle`, `keep_safe_distance` | 邻车确实从相邻车道切入；ego 及时制动且无碰撞 |
| E2 | 常规突发横穿 | 前方行人突然横穿，立即减速停车并注意避让 | `emergency` | `emergency_brake`, `yield_pedestrian`, `stop_safely` | 行人突然进入车道，留有正常紧急制动距离 |
| E2-Hard | 困难近距离横穿 | 行人近距离突然横穿，立即紧急制动并停车让行 | `emergency` | `hard_emergency_brake`, `yield_pedestrian`, `stop_safely` | 触发距离短于 E2，ego 需要更强制动但仍应避免碰撞 |
| E2-Critical | 临界距离横穿 | 行人在临界距离突然进入车道，立即最大制动并停车避让 | `emergency` | `maximum_emergency_brake`, `critical_pedestrian_avoidance`, `stop_safely` | 触发距离约 10–14 m，用于临界避碰；必须独立记录最小距离和碰撞结果 |
| E3 | 施工车道收窄 | 前方施工导致车道收窄，立即减速并安全并入左侧车道 | `emergency` | `decelerate`, `avoid_construction`, `merge_left` | 锥桶封闭 ego 路径；ego 减速并道、无碰撞，通过后继续行驶 |
| E4 | 前车突然急刹 | 前方车辆突然急刹，立即制动并保持安全距离 | `emergency` | `emergency_brake`, `avoid_vehicle`, `keep_safe_distance` | 前车产生明确速度骤降；ego 在响应窗口内制动且无追尾 |

## 4. 每帧样本的对齐规则

每个有效帧应至少保留以下内容：

```json
{
  "taxonomy_id": "E4",
  "scenario_name": "extreme_e4_lead_hard_brake",
  "command": {
    "command_text": "前方车辆突然急刹，立即制动并保持安全距离",
    "command_type": "emergency",
    "intent_label": [
      "emergency_brake",
      "avoid_vehicle",
      "keep_safe_distance"
    ],
    "target_speed_mps": 0.0,
    "target_lane": "current"
  },
  "event_phase": "response"
}
```

同一个 episode 可以使用同一条高层 command，但每帧还应通过 `event_phase` 区分：

| 阶段 | 含义 |
|---|---|
| `approach` | 危险或目标事件发生前，ego 正在接近 |
| `trigger` | 事件刚刚触发 |
| `response` | ego 正在制动、转向、变道或让行 |
| `recovery` | 事件完成后恢复行驶 |

这样可以避免“每帧 command 相同，但模型无法判断当前处于哪个动作阶段”的问题。

## 5. 自动验收与人工验收

### 自动验收

- 传感器时间同步；
- 图像、LiDAR、BEV 和轨迹文件完整；
- `sample_valid == true`；
- 场景事件触发并成功；
- 无不允许的碰撞；
- 速度、车道、转向和响应时间符合该小类配置。

### 人工验收

- command 描述的目标对象真实出现在画面中；
- 左转、右转和变道方向与文本一致；
- 拥堵车辆与 ego 路线相关；
- 行人和两轮车位于 ego 的有效交互区域；
- 应急行为不是普通巡航或人为制造的无关碰撞；
- 场景完成后的恢复阶段合理。

## 6. 当前数据迁移原则

当前已验收的 180 个样本先保持原始文件不变，作为可追溯基线。后续执行对齐时：

1. 从本文件生成统一的 taxonomy/command 映射；
2. 只处理通过验收的 episode；
3. 为每帧补充 `taxonomy_id` 和 `event_phase`；
4. 将 B5、B6 的 `command_type` 从现有 `compound` 规范化为 `single`；
5. 对其他 command 与意图标签进行批量一致性检查；
6. 输出新的训练索引，不覆盖原始标注。
