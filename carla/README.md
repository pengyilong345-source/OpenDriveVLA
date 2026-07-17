# OpenDriveVLA CARLA Data Pipeline

本目录包含两条互补的数据链路：

1. 将 Bench2Drive 官方专家数据筛选并转换为 150k 个 sample-v1.1 同步时间点。
2. 在 CARLA 0.9.15 中采集三类比赛场景的小规模自有样本，用于格式和场景适配验证。

Git 只保存代码、配置、筛选编号和格式说明，不保存 RGB、LiDAR、压缩包或转换后的数据集。

## 目录

```text
carla/
  bench2drive/   Bench2Drive 筛选、下载、转换、验证和官方评测接入
  collectors/   CARLA 自采与样本验证工具
  scenarios/    三类自采配置和 Bench2Drive 演示路线
  schema/       sample-v1.1 schema 与模板
```

## Bench2Drive 150k

目标分配：

| 类别 | 同步时间点 |
|---|---:|
| `basic_control` | 33,000 |
| `complex_obstacle_avoidance` | 62,000 |
| `extreme_emergency` | 55,000 |

完整的筛选依据、流式下载转换、断点续传和验收命令见
[`bench2drive/BENCH2DRIVE_150K_DATASET.md`](bench2drive/BENCH2DRIVE_150K_DATASET.md)。

Windows CARLA 与 WSL Bench2Drive 官方 evaluator 的接入方法见
[`BENCH2DRIVE_ADOPTION.md`](BENCH2DRIVE_ADOPTION.md)。

## CARLA 0.9.15 小规模自采

### 环境

- Windows CARLA Server：0.9.15
- WSL Ubuntu Python Client：0.9.15
- RPC 端口：2000

Windows PowerShell 启动 CARLA：

```powershell
$env:CARLA_ROOT = "C:\path\to\CARLA_0.9.15"
Set-Location $env:CARLA_ROOT
.\CarlaUE4.exe -quality-level=Epic -carla-rpc-port=2000
```

在 WSL 中进入项目并激活环境：

```bash
conda activate carla0915
cd /path/to/OpenDriveVLA
```

### 采集一组 v1.1 小样本

```bash
python carla/collectors/multimodal_collect.py \
  --config carla/scenarios/complex_obstacle_avoidance.pilot.json \
  --samples 5 \
  --vehicles 30 \
  --walkers 60 \
  --motorcycles 5 \
  --bicycles 5 \
  --seed 2201 \
  --output carla/output/v1_1_complex_pilot_01
```

验证输出：

```bash
python carla/collectors/validate_sample_v1_1.py \
  carla/output/v1_1_complex_pilot_01 \
  --expected-samples 5
```

### 分 episode 批量试采

```bash
EPISODES=1 SAMPLES_PER_EPISODE=5 \
  bash carla/collectors/collect_small_scale.sh complex
```

支持的类别参数为 `basic`、`complex`、`emergency` 和 `all`。正式运行前应先使用一个 episode 做链路检查。

## 自采数据的限制

当前自采脚本用于格式和传感器链路验证。随机 NPC 交通流不等价于比赛的受控事件，不能证明已经实现横穿行人、车辆加塞、前车急刹或施工封路。正式比赛数据仍需通过受控场景触发器和专家轨迹生成。

## 样本格式

sample-v1.1 包含：

- 六路同步 RGB
- LiDAR
- 相机内外参
- ego 状态与控制量
- 2 秒历史轨迹和 3 秒未来轨迹
- 3D actors
- 天气、地图和交通灯状态
- 场景模板指令及其来源声明

格式定义见 [`schema/README.md`](schema/README.md)。

## 提交规则

以下内容不得提交到 Git：

```text
carla/output/
**/__pycache__/
*.tar.gz
转换后的 RGB、LiDAR 和数据集目录
下载日志与临时缓存
```
