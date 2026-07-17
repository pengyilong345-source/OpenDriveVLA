# DriveVLA 下游控制模块 —— README

## 项目概述

本项目是 DriveVLA 多模态大模型的下游控制执行模块，核心功能是将上游规划层输出的 6 个未来轨迹点（覆盖未来 3 秒）实时转化为 CARLA 模拟器可执行的底层控制量：转向（Steer）、油门（Throttle）、刹车（Brake）。

该模块采用 纯跟踪（Pure Pursuit）横向控制 + PID 纵向控制 架构，同时集成安全监控模块，支持从真实数据集回放轨迹，并引入专家数据前馈控制。

## 文件目录结构

```
control_module/
├── carla_integration/          # CARLA 对接模块
│   ├── carla_client.py         # CARLA 客户端封装（连接、生成车辆、发送控制）
│   └── run_control_loop.py     # 主控制循环（数据集回放模式 + 专家前馈）
├── config/
│   └── default_config.yaml     # 配置文件（预留）
├── core/                       # 核心控制算法
│   ├── pid_controller.py       # 纵向 PID 控制器（含滑行优先策略）
│   └── trajectory_follower.py  # 横向纯跟踪控制器
├── safety/                     # 安全模块
│   └── safety_monitor.py       # 控制量限幅、变化率限制、紧急刹车
├── tools/                      # 分析与调试工具
│   ├── analysis.py             # 通用分析脚本（误差、路径对比）
│   ├── compare_steer.py        # 转向角对比（控制器 vs 专家）
│   ├── compare_velocity.py     # 速度跟踪分析
│   └── plot_dataset.py         # 数据集坐标可视化
├── utils/                      # 工具函数
│   ├── data_loader.py          # 数据集加载器（加载 frame_*.json）
│   ├── mock_data_generator.py  # Mock 数据生成器
│   └── trajectory_utils.py     # 轨迹插值、前瞻点计算
├── tests/                      # 测试
│   ├── test_offline.py         # 离线闭环测试
│   └── test_data/
│       └── mock_trajectories.json
├── HighwayExit_Town06_Route291_Weather5/   # 示例数据集
│   ├── annotations/            # 帧数据（frame_*.json）
│   ├── calib/                  # 相机标定文件
│   └── sensors/                # 传感器数据（图像、点云）
├── logs/                       # 运行日志（自动生成 CSV）
├── README.md
└── 自用记录.txt
```

## 环境要求

- Python 3.10（推荐使用 Conda 管理）
- CARLA 0.9.15（WindowsNoEditor 版本）
- 依赖包：
  ```bash
  pip install carla==0.9.15 pandas matplotlib numpy scipy
  ```

## 核心模块说明

### 1. 横向控制器（core/trajectory_follower.py）
- 算法：纯跟踪（Pure Pursuit）
- 核心公式：`steer = arctan(2 * y_lookahead / L^2 * wheelbase)`
- 可调参数：
  - `wheelbase`：车辆轴距（默认 2.8 m）
  - `lookahead_time`：前瞻时间（默认 1.2 秒，推荐范围 1.0 ~ 1.8）

### 2. 纵向控制器（core/pid_controller.py）
- 算法：增量式 PID，采用**滑行优先策略**
- 输出拆分：
  - `control > 0` → 油门
  - 轻微超速（误差 > -0.5 m/s）→ 滑行（油门刹车均为0）
  - 严重超速 → 按比例刹车，最大限幅 0.5
- 可调参数：
  - `kp`：比例增益（默认 0.6，建议 1.0~1.5）
  - `ki`：积分增益（默认 0.12）
  - `kd`：微分增益（默认 0.08）

### 3. 专家前馈控制（run_control_loop.py）
- 混合控制公式：`final_steer = expert_steer + correction_gain * (pure_pursuit_steer - expert_steer)`
- 以专家数据为主体，用纯跟踪做修正
- 可调参数：
  - `--alpha`：专家前馈权重（默认 0.7）
  - `--gain`：误差修正增益（默认 0.3）

### 4. 安全模块（safety/safety_monitor.py）
- 变化率限制：防止转向/油门/刹车突变
- 防冲突：油门和刹车不会同时输出
- 紧急刹车：高速突踩刹车时触发最大制动力

## 使用方法

### 1. 启动 CARLA 服务器

在 CARLA 根目录（如 `E:\course_file\OpenDriveVLA\CARLA_0.9.15\WindowsNoEditor`）下打开终端：

```bash
.\CarlaUE4.exe -dx11 -quality-level=Low
```

等待 CARLA 窗口出现地图画面。

### 2. 切换地图（如需要）

如果数据集要求的地图与当前不一致，使用 CARLA 官方工具切换：

```bash
cd PythonAPI\util
python config.py --map Town06
```

等待 5~15 秒，终端显示地图切换完成。

### 3. 运行控制循环（数据集回放模式）

在项目根目录下执行：

```bash
python control_module/carla_integration/run_control_loop.py --data_dir "E:\course_file\OpenDriveVLA\our_work\OpenDriveVLA\control_module\HighwayExit_Town06_Route291_Weather5\annotations" --duration 30.0
```

可选参数：
- `--duration`：运行时长（秒），默认 60.0
- `--alpha`：专家前馈权重（0~1），默认 0.7
- `--gain`：误差修正增益，默认 0.3

示例（调参）：
```bash
python run_control_loop.py --data_dir ./annotations --alpha 0.8 --gain 0.2
```

### 4. 在线运行模式（Mock 数据）

如果暂无数据集，可使用 Mock 数据测试：

```bash
python control_module/carla_integration/run_control_loop.py --data_dir ./tests/test_data
```

### 5. 离线测试（无需 CARLA）

验证控制器逻辑是否正确：

```bash
python control_module/tests/test_offline.py
```

生成 `logs/offline_test.csv` 日志文件。

## 分析工具

### 数据集可视化（Python）

检查数据集中 ego_state 与 future_trajectory 的坐标是否正确：

```bash
python control_module/tools/plot_dataset.py
```

- 红色路径：所有帧 ego_state 的位置连线
- 绿色散点：所有帧 future_trajectory 转换到世界坐标的点
- 彩色短线：前 30 帧的轨迹方向

### 转向角对比（控制器 vs 专家）

```bash
python control_module/tools/compare_steer.py
```

- 自动选择最新日志
- 输出：转向角对比图、误差统计（MAE、RMSE、最大误差）
- 判断：转向角是否与专家数据一致

### 速度跟踪分析

```bash
python control_module/tools/compare_velocity.py
```

- 输出：实际车速 vs 目标车速、油门/刹车输出、速度误差
- 诊断：油门饱和情况、加速能力是否充足

### 通用分析

```bash
python control_module/tools/analysis.py
```

- 横向误差、转向角、速度跟踪、路径对比
- 支持单个或对比多个日志文件

## 注意事项

### 1. 坐标系修正
数据集中 `future_trajectory_ego_frame` 的 `y` 坐标与 CARLA 世界坐标系可能存在左右反转。代码中已通过 `y = -p["y"]` 修正（见 `run_control_loop.py` 的 `_load_frame_trajectory` 方法），如方向仍不正确，可调整该符号。

### 2. 地图切换
- `load_world` 在 Windows 下不稳定，建议使用 `config.py` 切换地图
- 必须确保 CARLA 当前地图与数据集匹配（如 Town06）

### 3. 速度跟踪优化
- 车辆初始速度建议设置为数据集第一帧的速度（代码已实现）
- 刹车采用滑行优先策略，避免急刹导致车辆完全停止
- 如果速度仍然跟不上，可提高 PID 的 `kp`（建议 1.0~1.5）

### 4. 转向过猛
- 检查 `lookahead_time`：过大导致转向不足，过小导致转向过度
- 推荐值：1.0 ~ 1.5 秒，可根据车速动态调整

### 5. 日志说明
- 日志保存在 `logs/` 目录，文件名包含时间戳
- CSV 格式，可用 Excel、Python 或 MATLAB 打开分析
- 包含字段：帧索引、时间、位置、速度、控制量、专家转向角等

## 调参建议

| 问题 | 调整参数 | 方向 |
|------|----------|------|
| 转向不足（拐不过去） | `lookahead_time` | 减小（看近一点） |
| 转向过度（拐太猛） | `lookahead_time` | 增大（看远一点） |
| 加速太慢 | `kp`（PID） | 增大（1.0~1.5） |
| 刹车太猛 | `max_brake`（pid_controller） | 减小（0.3~0.5） |
| 与专家转向偏差大 | `--alpha` | 增大（0.8~0.9） |
| 转向抖动 | 安全模块 | 降低变化率限制 |

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: No module named 'carla'` | `pip install carla==0.9.15` |
| CARLA 窗口闪烁或崩溃 | 使用 `-dx11` 启动 |
| 车辆生成后不动 | 检查终端是否有报错；按 `Backspace` 重置 |
| 视角无法锁定车辆 | 鼠标点击车辆，按 `F` 键 |
| 地图切换失败 | 使用 `config.py --map Town06` |
| 数据集加载乱码 | `open()` 时指定 `encoding='utf-8'` |
| 车辆冲出主路 | 检查转向角符号；调整 `lookahead_time` |

## 下一步规划

### 近期
- 优化速度跟踪，使实际车速更接近专家数据
- 测试多场景数据集（不同城镇、天气）
- 建立量化评估指标体系（横向误差、速度误差、压线率）

### 中期
- 引入模型预测控制（MPC）替代纯跟踪
- 根据弯道曲率自动调整目标速度
- 生成自动化仿真测试报告

### 长期
- 端到端控制策略（模仿学习）
- 实车部署与验证

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-07-13 | 初始版本：纯跟踪 + PID + 安全模块 + CARLA 可视化 |
| v1.1 | 2026-07-15 | 增加数据集回放模式、动态边界、轨迹可视化 |
| v1.2 | 2026-07-16 | 增加专家前馈控制、速度对比分析、滑行优先刹车策略 |
| v1.3 | 2026-07-17 | 优化初始速度设置、完善分析工具、更新文档 |

## 联系方式

如有问题，请联系项目组成员或在项目群中讨论。

**最后更新**：2026 年 7 月 17 日