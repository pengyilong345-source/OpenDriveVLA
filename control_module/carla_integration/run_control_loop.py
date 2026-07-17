# """
# CARLA 控制主循环：加载真实数据集，按帧回放轨迹。
# 用法：
#     python run_control_loop.py --data_dir /path/to/annotations
# """
# import sys
# import os
# import time
# import csv
# import argparse
# import math
# from typing import Dict, List, Optional

# # 添加项目根目录到路径
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import carla
# import json

# from control_module.core.trajectory_follower import PurePursuitController
# from control_module.core.pid_controller import PIDController
# from control_module.safety.safety_monitor import SafetyMonitor
# from control_module.carla_integration.carla_client import CarlaClient
# from control_module.utils.trajectory_utils import interpolate_trajectory
# from control_module.utils.data_loader import load_frames


# class CarlaControlLoop:
#     def __init__(self, data_dir: str):
#         """
#         初始化，加载所有帧数据。
#         """
#         self.data_dir = data_dir
#         self.frames = load_frames(data_dir)
#         if not self.frames:
#             raise ValueError(f"在 {data_dir} 中没有找到任何 frame_*.json 文件")

#         print(f"✅ 加载了 {len(self.frames)} 帧数据")

#         # 动态边界
#         xs = [f["ego_state"]["x"] for f in self.frames]
#         ys = [f["ego_state"]["y"] for f in self.frames]
#         margin = 200.0
#         self.boundary_x_min = min(xs) - margin
#         self.boundary_x_max = max(xs) + margin
#         self.boundary_y_min = min(ys) - margin
#         self.boundary_y_max = max(ys) + margin
#         print(f"✅ 动态边界: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}]")

#         # 提取地图名称
#         self.map_name = self.frames[0].get("town", "Town06")
#         print(f"🗺️ 数据集地图: {self.map_name}")

#         # 初始化控制器
#         self.pid = PIDController(kp=0.6, ki=0.12, kd=0.08)
#         self.pure_pursuit = PurePursuitController(wheelbase=2.8, lookahead_time=1.2)
#         self.safety = SafetyMonitor()

#         # CARLA客户端
#         self.client = CarlaClient()

#         # 日志数据
#         self.log_data = []
#         self.log_file = None

#         # 帧索引
#         self.current_frame_idx = 0
#         self.last_frame_idx = len(self.frames) - 1

#         # 当前目标轨迹
#         self.current_trajectory = []

#         # 可视化开关
#         self.draw_enabled = True

#     def draw_trajectory_with_labels(self, vehicle_transform: carla.Transform,
#                                      trajectory_points: List[Dict],
#                                      color: carla.Color = carla.Color(0, 255, 0),
#                                      size: float = 0.3,
#                                      life_time: float = 0.1) -> None:
#         """
#         在CARLA世界中绘制目标轨迹点，并显示数字标签 [1] ~ [6]。
#         """
#         if not self.draw_enabled or not trajectory_points:
#             return
#         world = self.client.world
#         debug = world.debug

#         for idx, pt in enumerate(trajectory_points):
#             # 绘制点
#             local_location = carla.Location(x=pt['x'], y=pt['y'], z=0.1)
#             world_location = vehicle_transform.transform(local_location)
#             debug.draw_point(
#                 world_location,
#                 size=size,
#                 color=color,
#                 life_time=life_time
#             )

#             # 绘制标签
#             text_location = carla.Location(x=pt['x'], y=pt['y'], z=1.5)
#             text_world_location = vehicle_transform.transform(text_location)
#             debug.draw_string(
#                 text_world_location,
#                 f'  [{idx+1}]',
#                 draw_shadow=False,
#                 color=carla.Color(255, 255, 255),
#                 life_time=life_time
#             )

#     def run(self, duration_seconds: float = 60.0):
#         """
#         运行控制循环。
#         """
#         print("\n" + "=" * 60)
#         print("🚗 CARLA 控制循环 - 数据集回放模式")
#         print("=" * 60)

#         # 1. 连接CARLA
#         if not self.client.connect():
#             print("❌ 无法连接到CARLA，请检查服务器是否运行")
#             return

#         # 2. 校验地图是否匹配
#         current_map_name = self.client.world.get_map().name
#         if self.map_name not in current_map_name:
#             print(f"⚠️ 当前地图 {current_map_name} 与数据集要求 {self.map_name} 不匹配！")
#             print(f"请重启 CARLA 并指定地图：")
#             print(f"   .\\CarlaUE4.exe -dx11 -quality-level=Low -carla-map=/Game/Carla/Maps/{self.map_name}")
#             print(f"或使用 config.py 工具切换：")
#             print(f"   cd PythonAPI\\util")
#             print(f"   python config.py --map {self.map_name}")
#             return
#         else:
#             print(f"✅ 地图匹配: {current_map_name}")

#         # 3. 生成车辆
#         first_frame = self.frames[0]
#         ego = first_frame["ego_state"]
#         init_x = ego["x"]
#         init_y = ego["y"]
#         init_yaw = ego["yaw"]  # 弧度
#         init_transform = carla.Transform(
#             carla.Location(x=init_x, y=init_y, z=0.2),
#             carla.Rotation(yaw=math.degrees(init_yaw))
#         )
#         if not self.client.spawn_vehicle(spawn_point=init_transform):
#             print("❌ 车辆生成失败")
#             return

#         # 4. 日志文件
#         log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
#         os.makedirs(log_dir, exist_ok=True)
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         self.log_file = os.path.join(log_dir, f"replay_control_{timestamp}.csv")

#         print(f"日志将保存到: {self.log_file}")
#         print("-" * 60)
#         print("帧索引 | 时间(s) | 车速(m/s) | 油门 | 刹车 | 转向")
#         print("-" * 60)

#         # 5. 主循环
#         start_time = time.time()
#         last_print_time = start_time
#         step_count = 0
#         frame_update_time = 0.0

#         # 加载第一帧轨迹
#         self._load_frame_trajectory(0)

#         try:
#             while True:
#                 current_time = time.time() - start_time
#                 if current_time > duration_seconds:
#                     break

#                 # 获取车辆状态
#                 x, y, yaw, speed = self.client.get_vehicle_state()
#                 vehicle_transform = self.client.get_transform()

#                 # 边界检测
#                 if (x < self.boundary_x_min or x > self.boundary_x_max or
#                     y < self.boundary_y_min or y > self.boundary_y_max):
#                     print(f"⚠️ 车辆超出动态边界，终止 (x={x:.1f}, y={y:.1f})")
#                     break

#                 # 帧更新（使用数据中的 timestamp 计算间隔）
#                 if self.current_frame_idx < self.last_frame_idx:
#                     current_timestamp = self.frames[self.current_frame_idx]["timestamp"]
#                     next_timestamp = self.frames[self.current_frame_idx + 1]["timestamp"]
#                     frame_interval = max(0.01, next_timestamp - current_timestamp)
#                 else:
#                     frame_interval = 0.1

#                 if current_time - frame_update_time >= frame_interval:
#                     if self.current_frame_idx < self.last_frame_idx:
#                         self.current_frame_idx += 1
#                         self._load_frame_trajectory(self.current_frame_idx)
#                         frame_update_time = current_time

#                 # 控制计算
#                 if self.current_trajectory:
#                     steer, _ = self.pure_pursuit.compute_steer(self.current_trajectory, speed)
#                 else:
#                     steer = 0.0

#                 target_speed = self.frames[self.current_frame_idx]["ego_state"]["speed"]
#                 throttle, brake = self.pid.compute_throttle_brake(target_speed, speed)

#                 # 安全过滤
#                 steer, throttle, brake = self.safety.filter_control(steer, throttle, brake, speed)

#                 # 发送控制指令
#                 self.client.set_control(steer, throttle, brake)

#                 # 可视化轨迹点（当前帧的6个目标点）
#                 self.draw_trajectory_with_labels(
#                     vehicle_transform,
#                     self.current_trajectory,
#                     color=carla.Color(0, 255, 0),
#                     size=0.3,
#                     life_time=0.1
#                 )

#                 # 记录日志
#                 self.log_data.append({
#                     "frame_idx": self.current_frame_idx,
#                     "time": current_time,
#                     "x": x,
#                     "y": y,
#                     "speed": speed,
#                     "steer": steer,
#                     "throttle": throttle,
#                     "brake": brake,
#                     "yaw": yaw,
#                     "target_speed": target_speed,
#                 })

#                 # 打印状态
#                 if current_time - last_print_time >= 0.5:
#                     print(f"{self.current_frame_idx:6d} | {current_time:8.2f} | {speed:8.2f} | {throttle:6.3f} | {brake:6.3f} | {steer:6.3f}")
#                     last_print_time = current_time

#                 # 推进一帧
#                 self.client.step()
#                 step_count += 1

#         except KeyboardInterrupt:
#             print("\n⚠️ 用户中断")
#         except Exception as e:
#             print(f"❌ 运行时错误: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             # 清理
#             print("-" * 60)
#             print("正在停止车辆并保存日志...")
#             self.client.set_control(0.0, 0.0, 1.0)
#             time.sleep(0.5)
#             self.client.destroy()

#             # 保存日志
#             if self.log_data and self.log_file:
#                 with open(self.log_file, "w", newline="") as f:
#                     writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
#                     writer.writeheader()
#                     writer.writerows(self.log_data)
#                 print(f"✅ 日志已保存: {self.log_file}")

#             print(f"总步数: {step_count}")
#             print("=" * 60)

#     def _load_frame_trajectory(self, idx: int):
#         """
#         从指定帧加载 future_trajectory_ego_frame 作为当前目标轨迹。
#         """
#         frame = self.frames[idx]
#         traj = frame["future_trajectory_ego_frame"]
#         self.current_trajectory = [{"x": p["x"], "y": -p["y"]} for p in traj]


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="数据集回放控制")
#     parser.add_argument("--data_dir", type=str, required=True,
#                         help="包含 frame_*.json 的文件夹路径")
#     parser.add_argument("--duration", type=float, default=60.0,
#                         help="运行时长（秒）")
#     args = parser.parse_args()

#     loop = CarlaControlLoop(args.data_dir)
#     loop.run(duration_seconds=args.duration)


"""
CARLA 控制主循环：加载真实数据集，按帧回放轨迹
增强功能：
    - 引入专家转向角作为前馈
    - 从数据集中读取目标速度
    - 记录 expert_steer 供后续分析
用法：
    python run_control_loop.py --data_dir /path/to/annotations
"""
import sys
import os
import time
import csv
import argparse
import math
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import carla
import json

from control_module.core.trajectory_follower import PurePursuitController
from control_module.core.pid_controller import PIDController
from control_module.safety.safety_monitor import SafetyMonitor
from control_module.carla_integration.carla_client import CarlaClient
from control_module.utils.trajectory_utils import interpolate_trajectory
from control_module.utils.data_loader import load_frames


class CarlaControlLoop:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.frames = load_frames(data_dir)
        if not self.frames:
            raise ValueError(f"在 {data_dir} 中没有找到任何 frame_*.json 文件")

        print(f"✅ 加载了 {len(self.frames)} 帧数据")

        # 动态边界
        xs = [f["ego_state"]["x"] for f in self.frames]
        ys = [f["ego_state"]["y"] for f in self.frames]
        margin = 200.0
        self.boundary_x_min = min(xs) - margin
        self.boundary_x_max = max(xs) + margin
        self.boundary_y_min = min(ys) - margin
        self.boundary_y_max = max(ys) + margin
        print(f"✅ 动态边界: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}]")

        self.map_name = self.frames[0].get("town", "Town06")
        print(f"🗺️ 数据集地图: {self.map_name}")

        # ====== 控制器初始化 ======
        # 纵向控制：PID
        self.pid = PIDController(kp=0.6, ki=0.12, kd=0.08)
        # 横向控制：纯跟踪
        self.pure_pursuit = PurePursuitController(wheelbase=2.8, lookahead_time=1.2)
        # 安全模块
        self.safety = SafetyMonitor()

        # ====== 前馈参数 ======
        # alpha: 专家前馈权重 (0~1)，0 表示完全使用纯跟踪，1 表示完全使用专家数据
        # 推荐值：0.6~0.8，以专家为主，纯跟踪做修正
        self.feedforward_alpha = 0.7
        # 误差修正增益：对纯跟踪与专家偏差的修正力度
        self.correction_gain = 0.3

        # CARLA客户端
        self.client = CarlaClient()

        # 日志数据
        self.log_data = []
        self.log_file = None

        # 帧索引
        self.current_frame_idx = 0
        self.last_frame_idx = len(self.frames) - 1

        # 当前目标轨迹（自车坐标系），y 已取反修正
        self.current_trajectory = []

        # 可视化开关
        self.draw_enabled = True

    def draw_trajectory_with_labels(self, vehicle_transform: carla.Transform,
                                     trajectory_points: List[Dict],
                                     color: carla.Color = carla.Color(0, 255, 0),
                                     size: float = 0.3,
                                     life_time: float = 0.1) -> None:
        if not self.draw_enabled or not trajectory_points:
            return
        world = self.client.world
        debug = world.debug

        for idx, pt in enumerate(trajectory_points):
            local_location = carla.Location(x=pt['x'], y=pt['y'], z=0.1)
            world_location = vehicle_transform.transform(local_location)
            debug.draw_point(world_location, size=size, color=color, life_time=life_time)

            text_location = carla.Location(x=pt['x'], y=pt['y'], z=1.5)
            text_world_location = vehicle_transform.transform(text_location)
            debug.draw_string(
                text_world_location,
                f'  [{idx+1}]',
                draw_shadow=False,
                color=carla.Color(255, 255, 255),
                life_time=life_time
            )

    def compute_curvature_based_speed(self, trajectory, base_speed, max_speed=10.0, min_speed=2.0):
        """
        根据轨迹曲率动态调整目标速度。
        曲率越大，目标速度越低。
        """
        if not trajectory or len(trajectory) < 3:
            return base_speed

        # 取前几个点计算粗略曲率
        pts = trajectory[:4]
        # 计算相邻点之间的角度变化
        angles = []
        for i in range(1, len(pts) - 1):
            dx1 = pts[i]["x"] - pts[i-1]["x"]
            dy1 = pts[i]["y"] - pts[i-1]["y"]
            dx2 = pts[i+1]["x"] - pts[i]["x"]
            dy2 = pts[i+1]["y"] - pts[i]["y"]
            if dx1 == 0 and dy1 == 0:
                continue
            if dx2 == 0 and dy2 == 0:
                continue
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            # 角度变化
            dtheta = abs(angle2 - angle1)
            # 归一化到 [-pi, pi]
            dtheta = abs((dtheta + math.pi) % (2 * math.pi) - math.pi)
            angles.append(dtheta)

        if not angles:
            return base_speed

        avg_curvature = sum(angles) / len(angles)
        # 曲率越大，速度越低
        speed_factor = max(0.3, min(1.0, 1.0 - avg_curvature * 2.0))
        target_speed = max(min_speed, min(max_speed, base_speed * speed_factor))
        return target_speed

    def compute_steer_with_feedforward(self, pure_pursuit_steer, expert_steer):
        """
        专家前馈混合控制：
            final_steer = expert_steer * alpha + (pure_pursuit_steer - expert_steer) * correction_gain
            即：以专家为主，用纯跟踪的偏差做修正
        """
        # 计算偏差（纯跟踪与专家的差异）
        deviation = pure_pursuit_steer - expert_steer

        # 混合输出：专家数据为主体 + 偏差修正
        final_steer = expert_steer + self.correction_gain * deviation

        # 限幅
        final_steer = max(-1.0, min(1.0, final_steer))
        return final_steer

    def run(self, duration_seconds: float = 60.0):
        print("\n" + "=" * 60)
        print("🚗 CARLA 控制循环 - 数据集回放模式 (专家前馈)")
        print("=" * 60)

        # 1. 连接CARLA
        if not self.client.connect():
            print("❌ 无法连接到CARLA，请检查服务器是否运行")
            return

        # 2. 校验地图
        current_map_name = self.client.world.get_map().name
        if self.map_name not in current_map_name:
            print(f"⚠️ 当前地图 {current_map_name} 与数据集要求 {self.map_name} 不匹配！")
            print(f"请重启 CARLA 并指定地图：")
            print(f"   .\\CarlaUE4.exe -dx11 -quality-level=Low -carla-map={self.map_name}")
            return
        print(f"✅ 地图匹配: {current_map_name}")

        # 3. 生成车辆
        first_frame = self.frames[0]
        ego = first_frame["ego_state"]
        init_x = ego["x"]
        init_y = ego["y"]
        init_yaw = ego["yaw"]
        init_transform = carla.Transform(
            carla.Location(x=init_x, y=init_y, z=0.2),
            carla.Rotation(yaw=math.degrees(init_yaw))
        )
        if not self.client.spawn_vehicle(spawn_point=init_transform):
            print("❌ 车辆生成失败")
            return
        
        # ===== 新增：设置初始速度 =====
        initial_speed = first_frame["ego_state"]["speed"]  # 从数据集读取
        if initial_speed > 0.1:
            # 获取车辆朝向，将速度向量从局部坐标转换到世界坐标
            vehicle = self.client.vehicle
            transform = vehicle.get_transform()
            # 局部坐标系：x 向前，y 向左，z 向上
            local_velocity = carla.Vector3D(x=initial_speed, y=0.0, z=0.0)
            # 转换到世界坐标系
            world_velocity = transform.transform(local_velocity)
            vehicle.set_target_velocity(world_velocity)
            print(f"🚀 初始速度设置为: {initial_speed:.2f} m/s")

        # 4. 日志
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"replay_control_{timestamp}.csv")

        print(f"日志将保存到: {self.log_file}")
        print(f"前馈参数: alpha={self.feedforward_alpha}, correction_gain={self.correction_gain}")
        print("-" * 70)
        print("帧索引 | 时间(s) | 车速(m/s) | 油门 | 刹车 | 转向(控制器) | 转向(专家)")
        print("-" * 70)

        # 5. 主循环
        start_time = time.time()
        last_print_time = start_time
        step_count = 0
        frame_update_time = 0.0

        self._load_frame_trajectory(0)

        try:
            while True:
                current_time = time.time() - start_time
                if current_time > duration_seconds:
                    break

                x, y, yaw, speed = self.client.get_vehicle_state()
                vehicle_transform = self.client.get_transform()

                # 边界检测
                if (x < self.boundary_x_min or x > self.boundary_x_max or
                    y < self.boundary_y_min or y > self.boundary_y_max):
                    print(f"⚠️ 车辆超出动态边界，终止 (x={x:.1f}, y={y:.1f})")
                    break

                # ---- 帧更新 ----
                if self.current_frame_idx < self.last_frame_idx:
                    current_timestamp = self.frames[self.current_frame_idx]["timestamp"]
                    next_timestamp = self.frames[self.current_frame_idx + 1]["timestamp"]
                    frame_interval = max(0.01, next_timestamp - current_timestamp)
                else:
                    frame_interval = 0.1

                if current_time - frame_update_time >= frame_interval:
                    if self.current_frame_idx < self.last_frame_idx:
                        self.current_frame_idx += 1
                        self._load_frame_trajectory(self.current_frame_idx)
                        frame_update_time = current_time

                # 获取当前帧数据
                current_frame = self.frames[self.current_frame_idx]
                expert_steer = current_frame["ego_state"]["steer"]
                target_speed_base = current_frame["ego_state"]["speed"]

                # ---- 速度规划 ----
                # 方法1：直接使用数据集中的速度
                # target_speed = target_speed_base

                # 方法2：根据曲率动态调整（注释掉方法1，启用方法2）
                target_speed = self.compute_curvature_based_speed(
                    self.current_trajectory,
                    target_speed_base,
                    max_speed=10.0,
                    min_speed=2.0
                )

                # ---- 横向控制 ----
                if self.current_trajectory:
                    pure_pursuit_steer, lookahead = self.pure_pursuit.compute_steer(
                        self.current_trajectory, speed
                    )
                else:
                    pure_pursuit_steer = 0.0

                # 专家前馈混合
                steer = self.compute_steer_with_feedforward(pure_pursuit_steer, expert_steer)

                # ---- 纵向控制 ----
                throttle, brake = self.pid.compute_throttle_brake(target_speed, speed)

                # ---- 安全过滤 ----
                steer, throttle, brake = self.safety.filter_control(steer, throttle, brake, speed)

                # ---- 发送控制 ----
                self.client.set_control(steer, throttle, brake)

                # ---- 可视化 ----
                self.draw_trajectory_with_labels(vehicle_transform, self.current_trajectory)

                # ---- 日志 ----
                self.log_data.append({
                    "frame_idx": self.current_frame_idx,
                    "time": current_time,
                    "x": x,
                    "y": y,
                    "speed": speed,
                    "target_speed": target_speed,
                    "steer": steer,
                    "pure_pursuit_steer": pure_pursuit_steer,
                    "expert_steer": expert_steer,
                    "throttle": throttle,
                    "brake": brake,
                    "yaw": yaw,
                })

                # ---- 打印 ----
                if current_time - last_print_time >= 0.5:
                    print(f"{self.current_frame_idx:6d} | {current_time:8.2f} | {speed:8.2f} | "
                          f"{throttle:6.3f} | {brake:6.3f} | {steer:7.3f} | {expert_steer:7.3f}")
                    last_print_time = current_time

                self.client.step()
                step_count += 1

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("-" * 70)
            print("正在停止车辆并保存日志...")
            self.client.set_control(0.0, 0.0, 1.0)
            time.sleep(0.5)
            self.client.destroy()

            if self.log_data and self.log_file:
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.log_data)
                print(f"✅ 日志已保存: {self.log_file}")

            print(f"总步数: {step_count}")
            print("=" * 70)

    def _load_frame_trajectory(self, idx: int):
        """
        从指定帧加载 future_trajectory_ego_frame，并将 y 取反修正坐标系
        """
        frame = self.frames[idx]
        traj = frame["future_trajectory_ego_frame"]
        # y 取反：修正数据集与 CARLA 坐标系的左右差异
        self.current_trajectory = [{"x": p["x"], "y": -p["y"]} for p in traj]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集回放控制 (专家前馈)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="包含 frame_*.json 的文件夹路径")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="运行时长（秒）")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="专家前馈权重 (0~1)，默认 0.7")
    parser.add_argument("--gain", type=float, default=0.3,
                        help="误差修正增益，默认 0.3")
    args = parser.parse_args()

    loop = CarlaControlLoop(args.data_dir)
    loop.feedforward_alpha = args.alpha
    loop.correction_gain = args.gain
    loop.run(duration_seconds=args.duration)