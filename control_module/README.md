# DriveVLA 下游控制模块 —— README

## 项目概述

本项目是 **DriveVLA 多模态大模型** 的下游控制执行模块，核心功能是将上游规划层（DriveVLA / 路径规划模块）输出的 **6 个未来轨迹点（覆盖未来 3 秒）** 实时转化为 CARLA 模拟器可执行的底层控制量：**转向（Steer）、油门（Throttle）、刹车（Brake）**。

该模块采用经典的 **纯跟踪（Pure Pursuit）横向控制 + PID 纵向控制** 架构，同时集成安全监控模块，确保输出的控制量平滑、稳定、安全。

---

## 目录结构

```
control_module/
├── config/
│   └── default_config.yaml          # [预留] 控制器参数配置文件（后续启用）
├── core/
│   ├── __init__.py
│   ├── trajectory_follower.py       # 横向控制器：纯跟踪算法（Pure Pursuit）
│   └── pid_controller.py            # 纵向控制器：PID 速度控制
├── safety/
│   ├── __init__.py
│   └── safety_monitor.py            # 安全模块：控制量限幅、变化率限制、紧急刹车
├── utils/
│   ├── __init__.py
│   ├── trajectory_utils.py          # 轨迹插值、前瞻点计算、数据加载
│   └── mock_data_generator.py       # Mock 数据生成器（用于离线测试）
├── carla_integration/
│   ├── __init__.py
│   ├── carla_client.py              # CARLA 客户端封装（连接、生成车辆、发送控制）
│   └── run_control_loop.py          # CARLA 主控制循环（带轨迹可视化）
├── tests/
│   ├── __init__.py
│   ├── test_offline.py              # 离线闭环测试（不依赖 CARLA）
│   └── test_data/
│       └── mock_trajectories.json   # Mock 测试数据（6 个轨迹点）
├── logs/                             # [自动生成] 运行日志 CSV 文件
└── README.md                         # 本文档
```

---

## 环境要求与安装

### 1. 基础环境
- **Python 3.10**（推荐使用 Conda 管理）
- **CARLA 0.9.15**（需下载 WindowsNoEditor 版本）

### 2. 安装 CARLA Python API
```bash
# 激活你的 Conda 环境
conda activate drivevla

# 从 PyPI 安装 CARLA Python 客户端
pip install carla==0.9.15
```

### 3. 安装项目依赖（如需运行离线测试）
```bash
pip install matplotlib pyyaml
```

---

## 快速开始

### 模式一：离线测试（无需 CARLA 服务器）
验证控制器逻辑是否正确，使用 Mock 数据模拟车辆动力学。
```bash
cd E:\course_file\OpenDriveVLA\our_work\OpenDriveVLA
python control_module/tests/test_offline.py
```
**预期输出**：终端显示每 0.5 秒的车辆状态，最终生成 `logs/offline_test.csv` 日志文件。

---

### 模式二：CARLA 仿真测试（需要 CARLA 服务器）

#### 步骤 1：启动 CARLA 服务器
在 CARLA 安装目录（如 `E:\course_file\OpenDriveVLA\CARLA_0.9.15\WindowsNoEditor`）下打开终端：
```bash
.\CarlaUE4.exe -dx11 -quality-level=Low
```
等待窗口出现地图画面，服务器即就绪。

#### 步骤 2：运行控制循环
在项目根目录下执行：
```bash
python control_module/carla_integration/run_control_loop.py
```

**默认行为**：
- 使用 `tests/test_data/mock_trajectories.json` 作为轨迹输入。
- 在 CARLA 世界中生成一辆奥迪 TT，并按轨迹行驶。
- 在窗口中绘制轨迹点：
  - **绿色大点 + 数字标签 [1] ~ [6]**：原始 6 个规划点。
  - **黄色小点**：插值后的密集路径。
  - **红色大点**：当前前瞻点（车辆瞄准的目标）。

---

### 模式三：使用自定义轨迹文件
```bash
python control_module/carla_integration/run_control_loop.py /path/to/your/trajectory.json
```
轨迹文件格式要求：包含 `future_trajectory_ego_frame` 字段（6 个点，包含 `x`, `y` 坐标，可选 `dt` 时间戳）。

---

## 核心模块说明

### 1. 轨迹预处理（`utils/trajectory_utils.py`）
- `interpolate_trajectory()`：将 6 个稀疏点线性插值为密集点（默认步长 0.1 秒），使控制更平滑。
- `get_lookahead_point()`：根据当前车速动态计算前瞻点位置（前瞻距离 = 车速 × 前瞻时间）。

### 2. 横向控制器（`core/trajectory_follower.py`）
- **算法**：纯跟踪（Pure Pursuit）。
- **核心公式**：`steer = arctan(2 * y_lookahead / L^2 * wheelbase)`
- **可调参数**：
  - `wheelbase`：车辆轴距（默认 2.8 m）
  - `lookahead_time`：前瞻时间（默认 1.2 秒，推荐范围 1.0 ~ 1.8）

### 3. 纵向控制器（`core/pid_controller.py`）
- **算法**：增量式 PID。
- **输出拆分**：`control > 0` → 油门，`control < 0` → 刹车。
- **可调参数**：
  - `kp`：比例增益（默认 0.6）
  - `ki`：积分增益（默认 0.12）
  - `kd`：微分增益（默认 0.08）

### 4. 安全模块（`safety/safety_monitor.py`）
- **变化率限制**：防止转向/油门/刹车突变导致车辆抖动。
- **防冲突**：油门和刹车不会同时输出。
- **紧急刹车**：高速状态下突踩刹车时，自动触发最大制动力。

### 5. CARLA 客户端（`carla_integration/carla_client.py`）
- 封装 CARLA 的连接、车辆生成、状态获取、控制发送。
- 推荐使用 `-dx11` 启动 CARLA，以保证 Windows 下的稳定性。

---

## 接口定义（供上游调用）

### 统一调用接口（建议在 `interface.py` 中封装）
```python
from control_module.core.trajectory_follower import PurePursuitController
from control_module.core.pid_controller import PIDController
from control_module.safety.safety_monitor import SafetyMonitor

# 全局初始化（只执行一次）
follower = PurePursuitController(wheelbase=2.8, lookahead_time=1.2)
pid = PIDController(kp=0.6, ki=0.12, kd=0.08)
safety = SafetyMonitor()

def compute_control(trajectory_points: list, current_speed: float, target_speed: float = 6.0):
    """
    输入：
        trajectory_points: 6个点的列表，格式 [{"x": 1.0, "y": 0.0}, ...]
        current_speed: 当前车速 (m/s)
        target_speed: 目标车速 (m/s)，可选，默认 6.0
    
    输出：
        dict: {"steer": 0.1, "throttle": 0.4, "brake": 0.0}
    """
    steer, _ = follower.compute_steer(trajectory_points, current_speed)
    throttle, brake = pid.compute_throttle_brake(target_speed, current_speed)
    steer, throttle, brake = safety.filter_control(steer, throttle, brake, current_speed)
    return {"steer": steer, "throttle": throttle, "brake": brake}
```

---

## 注意事项

### 环境与操作
1. **CARLA 启动参数**：Windows 下必须使用 `-dx11`，`-RenderOffScreen` 可能在 Windows 上不稳定。
2. **路径要求**：CARLA 安装路径和项目路径**不可包含中文字符**，否则 Unreal Engine 可能加载失败。
3. **Python 版本**：必须使用 **Python 3.10**（与 PyTorch 2.1.2 和 CARLA 0.9.15 兼容）。

### 常见问题
| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: No module named 'carla'` | 执行 `pip install carla==0.9.15` |
| CARLA 窗口闪烁或崩溃 | 使用 `-dx11` 启动，或降低画质 `-quality-level=Low` |
| 车辆生成后不动 | 检查终端是否报错；按 `Backspace` 重置车辆位置 |
| 视角无法锁定车辆 | 用鼠标左键单击车辆，再按 `F` 键 |

### 已知限制
- **当前轨迹来源**：`mock_trajectories.json` 由数学公式生成，**不代表真实车道线**。弯道场景下可能出现压线，属于数据偏差，非控制器故障。
- **纵向控制**：目前使用固定目标速度（来自 JSON 中的 `ego_state.speed`），尚未集成速度规划模块。

---

## 改进意见

### 短期（可立即执行）
1. **车道偏离抑制**：在安全模块中增加横向误差反馈，轻微修正转向角，减少压线概率。
2. **参数自适应**：根据车速动态调整前瞻时间（低速增大、高速减小），提升弯道跟踪精度。

### 中期
1. **模型预测控制（MPC）替代纯跟踪**：MPC 可同时考虑车辆动力学约束和执行器延迟，适合高速/复杂场景。
2. **速度规划集成**：根据前方曲率自动计算目标速度（弯道减速、直道加速）。

### 长期
1. **端到端控制策略**：使用大规模采集的“轨迹-控制量”数据，训练一个小型神经网络，替代传统控制算法，实现更拟人的驾驶风格。

---

## 下一步规划

### 近期目标（1-2 周）
- [ ] 对接数据集同学，获取真实采集数据（包含 `future_trajectory_ego_frame` 和 `steer/throttle/brake` 标签）。
- [ ] 用真实数据替换 Mock 数据，验证控制器在真实场景下的表现。
- [ ] 调参并记录各项指标（横向误差、速度误差、压线率）。

### 中期目标（1 个月）
- [ ] 实现在 CARLA 中连续多场景的自动评测（含弯道、环岛、跟车等场景）。
- [ ] 生成仿真测试报告，作为项目交付物之一。

### 长期目标（2-3 个月）
- [ ] 探索基于学习的控制策略（模仿学习 / 强化学习），提升复杂场景的适应性。

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-07-13 | 初始版本，完成纯跟踪 + PID + 安全模块 + CARLA 可视化集成 |
| v1.1 | 2026-07-14 | 规划中：添加车道偏离抑制，优化弯道跟踪性能 |

---

## 联系方式

如有问题，请联系项目组成员或在项目群中讨论。

**最后更新**：2026 年 7 月 13 日

---
