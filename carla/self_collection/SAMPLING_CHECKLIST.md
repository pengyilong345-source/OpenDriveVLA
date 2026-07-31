# CARLA 自采样执行清单

本清单用于 CARLA 0.9.15 自采样。必须先完成最小闭环，再开始三类场景批量采样。

## 一、环境检查

- [ ] Windows 中确认 CARLA 安装文件存在：`D:\Program\CARLA\CARLA_0.9.15\CarlaUE4.exe`
- [ ] 进入 Ubuntu：`wsl -d ubuntu`
- [ ] 激活环境：`conda activate carla0915`
- [ ] 确认终端提示符显示 `(carla0915)`
- [ ] 验证 Client：`python -c "import carla; print('CARLA Client OK')"`
- [ ] 进入 OpenDriveVLA 仓库根目录

```bash
cd "/path/to/OpenDriveVLA"
```

## 二、启动 CARLA 服务端

- [ ] 在 Windows PowerShell 中进入 CARLA 目录
- [ ] 使用低画质和 RPC 端口 2000 启动模拟器
- [ ] 等待城市画面加载完成
- [ ] 保持 CARLA 窗口打开
- [ ] 确认没有其他采样程序占用同步模式

```powershell
cd D:\Program\CARLA\CARLA_0.9.15
.\CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000
```

## 三、最小闭环试采

- [ ] 确认输出目录 `smoke_minimal_001` 不存在或为空
- [ ] 使用 8 辆普通车、3 名行人采集 2 个样本
- [ ] 观察 ego 是否正常生成并启用自动驾驶
- [ ] 观察 NPC 和行人是否正常生成
- [ ] 确认程序正常结束且无连接超时
- [ ] 确认 CARLA 中的 ego、NPC、行人和传感器已清理

```bash
python carla/self_collection/collectors/minimal_collect.py \
  --samples 2 \
  --vehicles 8 \
  --walkers 3 \
  --output "/path/to/carla_self_collection/smoke_minimal_001"
```

### 最小样本检查

- [ ] 输出目录包含 2 个样本目录
- [ ] 每个样本包含六路 RGB 图像
- [ ] 六路图像属于同一个 CARLA frame
- [ ] 图像可以正常打开，且不是全黑、损坏或严重错位
- [ ] 元数据包含 frame、时间戳、ego 位姿、速度和控制量
- [ ] 周围车辆与行人信息可以正常读取

## 四、完整 sample-v1.1 试采

### 基础控制

- [ ] 使用 8 辆车、3 名行人运行 1 个 episode
- [ ] 每个 episode 采集 5 个样本
- [ ] 自动验证结果为 PASS
- [ ] 人工抽查六路 RGB、LiDAR、BEV 和标注

```bash
OUTPUT_ROOT="/path/to/carla_self_collection/basic_control" \
EPISODES=1 \
SAMPLES_PER_EPISODE=5 \
bash carla/self_collection/scripts/collect_small_scale.sh basic
```

### 复杂避障

- [ ] 使用 18 辆车、10 名行人、2 辆摩托车、2 辆自行车
- [ ] 运行 1 个 episode，每个 episode 采集 5 个样本
- [ ] 自动验证结果为 PASS
- [ ] 检查关键交通参与者是否进入 ego 的有效感知范围
- [ ] 检查行人和两轮车没有只生成在 ego 后方或过远位置

```bash
OUTPUT_ROOT="/path/to/carla_self_collection/complex_obstacle" \
EPISODES=1 \
SAMPLES_PER_EPISODE=5 \
bash carla/self_collection/scripts/collect_small_scale.sh complex
```

### 极限应急

- [ ] 使用 10 辆车、2 名行人运行 1 个 episode
- [ ] 每个 episode 采集 5 个样本
- [ ] 自动验证结果为 PASS
- [ ] 检查大雨、低能见度和应急指令是否正确写入标注
- [ ] 检查背景车流没有遮挡或干扰关键危险事件
- [ ] 检查 `event.type` 为 `lead_vehicle_hard_brake`
- [ ] 检查 `event.triggered` 和 `event.success` 均为 `true`
- [ ] 检查 `event.event_actor_id`、触发帧和最小距离均已记录
- [ ] 检查 ego 出现可测制动或纵向减速度
- [ ] 检查 `event.collision` 为 `false`

```bash
OUTPUT_ROOT="/path/to/carla_self_collection/extreme_emergency" \
EPISODES=1 \
SAMPLES_PER_EPISODE=5 \
bash carla/self_collection/scripts/collect_small_scale.sh emergency
```

### 极限应急：相邻车切入

- [ ] 使用 10 辆背景车、2 名行人运行 1 个 episode
- [ ] 每个 episode 采集 5 个样本，自动验证结果为 PASS
- [ ] 检查 `event.type` 为 `adjacent_vehicle_cut_in`
- [ ] 检查 `event.triggered`、`event.state = completed` 和 `event.success` 均成立
- [ ] 检查场景车确实进入 ego 车道，ego 出现可测制动或纵向减速度
- [ ] 检查 `event.collision` 为 `false`

```bash
OUTPUT_ROOT="/path/to/data/carla_self_collection/extreme_cut_in" \
EPISODES=1 \
SAMPLES_PER_EPISODE=5 \
bash carla/self_collection/scripts/collect_small_scale.sh emergency_cutin
```

## 五、每个 Episode 的质量检查

- [ ] 记录随机种子、地图、天气、场景类别和子场景类型
- [ ] 记录请求生成和实际生成的车辆数
- [ ] 记录请求生成和实际生成的行人数
- [ ] 记录请求生成和实际生成的摩托车、自行车数
- [ ] 记录 ego 配置半径内的实际 Actor 数
- [ ] 检查六路 RGB 帧号一致
- [ ] 检查 RGB、LiDAR、BEV、标注和轨迹数量一致
- [ ] 检查历史 2 秒和未来 3 秒轨迹完整
- [ ] 检查控制量包含 throttle、steer、brake
- [ ] 检查没有传感器超时、丢帧或空文件
- [ ] 检查没有与目标无关的碰撞、卡死或路线失败
- [ ] 确认验证脚本返回 PASS
- [ ] 对失败 episode 记录原因，不混入有效数据集

## 六、扩量前验收

- [ ] 基础控制连续 3 个 episode 通过
- [ ] 复杂避障连续 3 个 episode 通过
- [ ] 极限应急连续 3 个 episode 通过
- [ ] 中断采样后 CARLA 可以恢复到异步模式
- [ ] 连续运行后没有残留 Actor 或传感器
- [ ] 输出目录位于 Git 仓库外
- [ ] 磁盘剩余空间满足计划采样量
- [ ] 随机种子规则与交通密度档位已固定
- [ ] 低、中、高密度均完成至少一次验证
- [ ] 团队确认样本字段满足训练数据需求

## 七、Git 提交前检查

- [ ] 只提交代码、场景配置、Schema 和文档
- [ ] 不提交 RGB、LiDAR、BEV、压缩包、日志和缓存
- [ ] `git status` 中没有采样数据文件
- [ ] Python、JSON 和 Bash 静态检查通过
- [ ] README 中的启动命令与实际路径一致
- [ ] 在 `carla_data_collector` 分支创建清晰的提交
- [ ] 推送分支后再发起代码审查

## 八、正式采样

- [ ] 按 22.0% / 41.3% / 36.7% 分配基础、复杂、应急数据量
- [ ] 每类场景覆盖低、中、高交通密度
- [ ] 每个 episode 使用独立且可复现的随机种子
- [ ] 定期备份进度清单和验证结果
- [ ] 每完成一批数据就进行自动验证和人工抽检
- [ ] 在达到目标规模前持续监控磁盘、FPS、失败率和类别分布
