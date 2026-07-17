# OpenDriveVLA CARLA Data Pipeline

当前仓库只发布经过检查的 Bench2Drive 数据筛选、下载、流式转换和验证链路。
CARLA 自采样脚本与自采场景配置仍在本地审查阶段，暂不纳入 `main`。

Git 仅保存代码、筛选清单、格式定义和文档，不保存 RGB、LiDAR、下载压缩包、
转换后的数据集、缓存或运行日志。

## 目录

```text
carla/
  bench2drive/  Bench2Drive 筛选、下载、转换、验证和官方评测接入
  scenarios/    Bench2Drive 候选路线
  schema/       sample-v1.1 schema 与模板
```

## Bench2Drive 150k

| 类别 | 同步时间点 |
|---|---:|
| `basic_control` | 33,000 |
| `complex_obstacle_avoidance` | 62,000 |
| `extreme_emergency` | 55,000 |
| **总计** | **150,000** |

完整的筛选依据、分类参数、流式转换、断点续传和验收命令见
[`bench2drive/BENCH2DRIVE_150K_DATASET.md`](bench2drive/BENCH2DRIVE_150K_DATASET.md)。

Windows CARLA 与 WSL Bench2Drive 官方 evaluator 的接入方法见
[`BENCH2DRIVE_ADOPTION.md`](BENCH2DRIVE_ADOPTION.md)。

## 验证结果

本地转换数据已完成全量结构检查和 RGB 实际解码：

| 类别 | 样本 | RGB 解码 | LiDAR | 错误 |
|---|---:|---:|---:|---:|
| 基础操控 | 33,000 | 198,000 | 33,000 | 0 |
| 复杂避障 | 62,000 | 372,000 | 62,000 | 0 |
| 极限应急 | 55,000 | 330,000 | 55,000 | 0 |
| **总计** | **150,000** | **900,000** | **150,000** | **0** |

## 样本格式

sample-v1.1 包含六路同步 RGB、LiDAR、相机标定、ego 状态与控制量、
2 秒历史轨迹、3 秒未来轨迹、3D actors、天气、地图状态和场景模板指令。
格式定义见 [`schema/README.md`](schema/README.md)。

## 提交边界

以下内容不得提交到 Git：

```text
carla/output/
**/__pycache__/
*.tar.gz
转换后的 RGB、LiDAR 和数据集目录
下载日志与临时缓存
尚未审查的 CARLA 自采样脚本和场景配置
```
