# OpenDriveVLA CARLA Sample Schema v1.1

本目录定义 CARLA 自采数据与 Bench2Drive 转换数据共同采用的同步样本格式。

## 文件

- `sample_v1_1.schema.json`：JSON Schema Draft 2020-12 约束。
- `sample_v1_1.template.json`：一条复杂避障同步样本示例。

## 核心约定

- `schema_version` 固定为 `1.1.0`。
- CARLA 版本为 `0.9.15`。
- 每条样本对应一个同步时间点，包含六路 RGB、LiDAR、ego 状态、轨迹、actors、天气、标定和指令。
- 速度统一使用 m/s，角度使用 rad，长度使用 m。
- 历史轨迹覆盖 `[-2.0, -1.5, -1.0, -0.5, 0.0] s`。
- 未来轨迹覆盖 `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0] s`。
- 轨迹、actor 相对位置和 3D bbox 使用当前 ego 坐标系：x 向前、y 向左、z 向上。
- `bbox_3d` 顺序为 `[center_x, center_y, center_z, length, width, height, yaw]`。
- `track_id` 在同一 episode 内保持稳定。

## LiDAR

自采 CARLA 数据：

```text
format: .bin
dtype: float32
fields: [x, y, z, intensity]
coordinate: current ego frame
```

Bench2Drive 官方数据：

```text
format: .laz
fields: [x, y, z]
coordinate: current ego frame
```

官方 `.laz` 不伪造不存在的 intensity。

## Episode 结构

```text
episode_name/
  annotations/frame_000120.json
  sensors/frame_000120/
    front.jpg
    front_left.jpg
    front_right.jpg
    rear.jpg
    rear_left.jpg
    rear_right.jpg
    lidar.bin|lidar.laz
  calib/
    camera_intrinsics.json
    camera_extrinsics.json
```

标定文件按 episode 共享，JSON 中的文件路径均相对于 episode 根目录。

## 场景类别

```text
1 = basic_control
2 = complex_obstacle_avoidance
3 = extreme_emergency
```

详细字段与取值限制以 `sample_v1_1.schema.json` 为准。
