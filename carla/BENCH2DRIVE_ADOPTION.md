# Bench2Drive 接入路线

本文档记录 OpenDriveVLA 对 Bench2Drive 的参考边界、环境差异和最小接入顺序。现阶段停止扩大随机 Traffic Manager 数据，优先打通一个官方受控场景。

## 外部参考

- 官方源码位于工作区 `resource/Bench2Drive`，仅作为外部参考，不纳入 OpenDriveVLA 仓库。
- Bench2Drive 主仓库使用 CC BY-NC-ND 4.0；不得直接复制并修改其受该许可约束的代码后再作为本项目代码发布。
- 仓库内 ScenarioRunner 目录使用 MIT License。后续优先依赖上游 ScenarioRunner，或将官方仓库作为外部运行依赖，不复制 Bench2Drive 私有实现。
- 不下载 Base/Full 数据集。Mini 也只在需要核对真实文件格式时再考虑下载。

## 官方实现审计

Bench2Drive 的关键数据约定：

- CARLA 文档配置为 0.9.15，采集帧率为 10 Hz。
- 数据单元是绑定 `scenario_name`、Town、weather 和 route 的短 clip，不是随机 NPC episode。
- 六路 RGB 为 1600 x 900；前/侧相机 FOV 70，后相机 FOV 110；JPG quality 20。
- LiDAR 为 64 线、85 m、600000 points/s、10 Hz，保存为 LAZ。
- 另有六路 depth、semantic、instance，相同相机位姿；还有 5 路 radar、IMU、GNSS、top-down debug camera 和 expert assessment。
- annotation 使用 gzip JSON，直接记录世界坐标、控制量、导航 command、传感器标定和完整 bounding boxes。
- 目录按 `scenario_name/Town_weather_route` 组织，各传感器按类型建立目录，帧号在 clip 内连续。

已观察到的兼容性风险：

- README 指向 CARLA 0.9.15，但仓库内 `scenario_runner/CARLA_VER` 仍为 0.9.13；以 README 和实际 Python API 兼容测试为准。
- 官方训练/验证路线大量使用 Town12/Town13，需要 CARLA 0.9.15 Additional Maps。当前 Windows 安装目录未发现明确的 Town12/Town13 资源，必须通过 `client.get_available_maps()` 再确认。
- 官方 evaluator 默认自行启动 Linux `CarlaUE4.sh`；本项目是 Windows CARLA 服务端加 WSL 客户端，必须改为连接外部服务端，不能直接使用官方启动脚本。
- 官方依赖固定在较老版本，例如 py-trees 0.8.3、numpy 1.18.4 和 Shapely 1.7.1。应创建独立环境验证，不能直接污染现有 `carla0915`。

## 第一阶段：只打通一个官方场景

首选 `DynamicObjectCrossing` 或 `VehicleTurningRoutePedestrian`，原因是它们能验证行人/骑行者触发、ego 制动、actor 标注和事件成功判定。

完成标准：

1. Windows CARLA 0.9.15 服务端可以加载目标 Town。
2. WSL 中的 ScenarioRunner 可以连接 Windows host:2000。
3. 使用一条裁剪后的官方 route，仅运行一个受控 scenario。
4. 专家 agent 可以完成路线且不碰撞。
5. 场景确实触发；失败运行不得写成成功样本。
6. 保存 10 Hz 连续 clip，并保留完整事件前、中、后阶段。
7. 将官方原始 annotation 转换为项目 v1.1 格式作为派生数据，不修改官方原始输出。

## 推荐执行顺序

### 1. 地图与 API 预检

从 WSL 连接已启动的 Windows CARLA：

```bash
HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
python -c "import carla; c=carla.Client('$HOST_IP',2000); c.set_timeout(20); print(c.get_server_version()); print('\n'.join(c.get_available_maps()))"
```

若列表没有 Town12/Town13，先安装 CARLA 0.9.15 Additional Maps；在此之前不要尝试官方 training/validation route。

### 2. 独立 Bench2Drive 环境

不要立即安装依赖。先根据 Python 3.8 和 CARLA 0.9.15 解析依赖冲突，再创建独立 conda 环境 `bench2drive0915`。现有 `carla0915` 继续用于项目采集器。

外部仓库和 CARLA PythonAPI 不写死在脚本中。运行官方评测相关脚本前设置：

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export CARLA_PYTHONAPI=/path/to/CARLA_0.9.15/PythonAPI/carla
```

### 3. 外部服务端适配

使用 `carla/bench2drive/run_external_evaluator.sh`，让官方评测器连接已经在
Windows 中运行的 CARLA 0.9.15。默认预检使用官方 Dev10 路线文件中的
Town12 路线 `2091` 和官方 NPC agent，结果写入
`carla/output/bench2drive_preflight`。

```bash
conda activate bench2drive0915
cd /path/to/OpenDriveVLA
bash carla/bench2drive/run_external_evaluator.sh
```

新增项目侧 launcher，负责设置以下变量并调用外部 Bench2Drive evaluator：

```text
CARLA_HOST=<Windows host IP>
PORT=2000
TM_PORT=8000
SCENARIO_RUNNER_ROOT=<resource/Bench2Drive/scenario_runner>
LEADERBOARD_ROOT=<resource/Bench2Drive/leaderboard>
```

launcher 不启动或关闭 Windows CARLA，只负责连接。

### 4. 单场景 route

从官方 XML 中选择一个与本机地图资源匹配的 route，保留短路线和一个 scenario。不得先运行包含大量场景的完整 `routes_training.xml`。

### 5. 原始数据与派生数据分层

```text
carla/output/bench2drive_raw/<clip>/       # 官方风格原始输出
carla/output/bench2drive_v1_1/<clip>/      # 转换后的项目格式
```

原始数据保持可追溯；转换器记录源 route、scenario、frame 和转换版本。

## 暂不执行

- 不继续生成 100 个高密度随机 episode。
- 不把随机交通流标成复杂避障。
- 不一次启用官方全部传感器；第一条链路先使用六路 RGB、LiDAR 和 annotation，确认显存后再增加 depth/semantic/instance/radar。
- 不启动 4 TB Full 或 400 GB Base 数据下载。
- 不在格式和场景成功判定未稳定前讨论 200k 帧正式采集。
