
# #!/usr/bin/env python
# """
# 使用 OpenDriveVLA 预测轨迹在 CARLA 中仿真
# 支持两种横向控制器：
#   - pure_pursuit: 纯跟踪 (基线)
#   - mpc: 模型预测控制 (优化)
# 用法：
#     python run_prediction_in_carla.py --json /path/to/predictions.json [--duration 30] [--controller mpc]
# """
# import sys
# import os
# import json
# import time
# import argparse
# import math
# import csv

# # 添加项目根目录到路径
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import carla

# from control_module.core.trajectory_follower import PurePursuitController
# from control_module.core.pid_controller import PIDController
# from control_module.safety.safety_monitor import SafetyMonitor
# from control_module.carla_integration.carla_client import CarlaClient
# from control_module.core.mpc_controller import MPCController


# class PredictionReplayController:
#     def __init__(self, json_path, controller_type="pure_pursuit"):
#         with open(json_path, 'r') as f:
#             self.data = json.load(f)
#         self.samples = self.data.get("samples", [])
#         self.current_idx = 0
#         self.total = len(self.samples)
#         print(f"加载 {self.total} 帧预测数据")
        
#         # 计算数据总时长（用于自动确定仿真时间）
#         if self.total > 0:
#             first_sim_t = self.samples[0].get("sim_t", 0.0)
#             last_sim_t = self.samples[-1].get("sim_t", 0.0)
#             # 每帧预测覆盖未来 3 秒（最后一个点的 dt 约为 3.0）
#             self.total_data_duration = (last_sim_t - first_sim_t) + 3.0
#             # 加一点余量
#             self.total_data_duration = max(5.0, self.total_data_duration + 1.0)
#             print(f"自动计算数据总时长: {self.total_data_duration:.2f} 秒")
#         else:
#             self.total_data_duration = 30.0
        
#         # 控制器初始化
#         self.controller_type = controller_type
#         if controller_type == "mpc":
#             self.mpc = MPCController(wheelbase=2.5, dt=0.1, N=15)
#             print("✅ 使用 MPC 横向控制器")
#         else:
#             self.pure_pursuit = PurePursuitController(wheelbase=2.5, lookahead_time=0.8)
#             print("✅ 使用 Pure Pursuit 横向控制器")
        
#         self.pid = PIDController(kp=2.0, ki=0.15, kd=0.1)
#         self.safety = SafetyMonitor()
#         self.client = CarlaClient()
#         self.log_data = []
        
#         self.current_traj = None
#         self.target_speed = 3.0
#         self.frame_update_time = 0.0
#         self.frame_interval = 0.1  # 每帧之间的时间间隔（从 sim_t 计算更准确）
        
#         # 新增：标记是否已经完成所有帧的加载
#         self.all_frames_loaded = False
#         # 记录最后一帧的加载时间，用于计算延续时长
#         self.last_frame_load_time = 0.0

#     def run(self, duration=None, start_speed=0.0):
#         # 如果未指定 duration，使用自动计算的时长
#         if duration is None:
#             duration = self.total_data_duration
#             print(f"使用自动时长: {duration:.2f} 秒")
#         else:
#             print(f"使用用户指定时长: {duration:.2f} 秒")
        
#         if not self.client.connect():
#             print("❌ 无法连接 CARLA")
#             return
        
#         init_transform = carla.Transform(
#             carla.Location(x=0, y=0, z=0.2),
#             carla.Rotation(yaw=90)
#             # 车头朝向 y 方向，便于与预测轨迹对齐
#             # 如果是yaw=0，车头朝向 x 方向
#         )
#         if not self.client.spawn_vehicle(spawn_point=init_transform):
#             print("❌ 车辆生成失败")
#             return
        
#         if start_speed > 0.1:
#             vehicle = self.client.vehicle
#             transform = vehicle.get_transform()
#             local_vel = carla.Vector3D(x=start_speed, y=0, z=0)
#             world_vel = transform.transform(local_vel)
#             vehicle.set_target_velocity(world_vel)
        
#         log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
#         os.makedirs(log_dir, exist_ok=True)
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         log_path = os.path.join(log_dir, f"prediction_replay_{timestamp}.csv")
#         print(f"日志将保存至 {log_path}")
        
#         start_time = time.time()
#         step_count = 0
#         current_time = 0.0
#         frame_update_time = 0.0
        
#         # 记录第一帧的时间
#         first_sim_t = self.samples[0].get("sim_t", 0.0) if self.total > 0 else 0.0
#         last_sim_t = self.samples[-1].get("sim_t", 0.0) if self.total > 0 else 0.0
        
#         # 帧间隔：如果数据帧之间时间差很小，用 0.1s；否则用实际差值
#         if self.total > 1:
#             self.frame_interval = max(0.05, (last_sim_t - first_sim_t) / (self.total - 1))
#         else:
#             self.frame_interval = 0.1
        
#         try:
#             while current_time < duration:
#                 current_time = time.time() - start_time
                
#                 x, y, yaw, speed = self.client.get_vehicle_state()
#                 vehicle_transform = self.client.vehicle.get_transform()
#                 ego_x = vehicle_transform.location.x
#                 ego_y = vehicle_transform.location.y
#                 ego_yaw = math.radians(vehicle_transform.rotation.yaw)
                
#                 # ---- 更新预测帧 ----
#                 if not self.all_frames_loaded and self.current_idx < self.total:
#                     # 根据实际 sim_t 差值来确定帧更新时机
#                     if self.current_idx == 0:
#                         # 第一帧立即加载
#                         self._load_frame(self.current_idx)
#                         self.current_idx += 1
#                         frame_update_time = current_time
#                         self.last_frame_load_time = current_time
#                     else:
#                         # 检查是否到了加载下一帧的时间
#                         if self.current_idx < self.total:
#                             # 计算当前帧与下一帧的 sim_t 差值
#                             next_sim_t = self.samples[self.current_idx].get("sim_t", 0.0)
#                             # 相对于当前时间，该帧应该在什么时候加载？
#                             # 我们使用绝对时间差：从第一帧开始累计
#                             first_sim_t = self.samples[0].get("sim_t", 0.0)
#                             expected_time = (next_sim_t - first_sim_t)  # 相对起始时间
#                             # 如果当前时间已经超过 expected_time，加载下一帧
#                             if current_time >= expected_time:
#                                 self._load_frame(self.current_idx)
#                                 self.current_idx += 1
#                                 frame_update_time = current_time
#                                 self.last_frame_load_time = current_time
                
#                 # ---- 如果所有帧已加载完，继续使用最后一帧的轨迹 ----
#                 # 但我们不立即清空轨迹，而是继续跟踪最后一帧的轨迹
#                 # 直到超出最后一帧的预测时间（3秒）后再停车
#                 if self.all_frames_loaded:
#                     # 检查从加载最后一帧到现在是否已经超过 3 秒
#                     if current_time - self.last_frame_load_time > 3.0:
#                         # 超出预测范围，可以停止跟踪
#                         self.current_traj = None
#                         self.target_speed = 0.0
                
#                 # ---- 控制计算 ----
#                 if self.current_traj is not None:
#                     if self.controller_type == "mpc":
#                         steer = self.mpc.compute_steer(
#                             self.current_traj,
#                             speed,
#                             ego_x,
#                             ego_y,
#                             ego_yaw
#                         )
#                     else:
#                         steer, _ = self.pure_pursuit.compute_steer(self.current_traj, speed)
#                 else:
#                     steer = 0.0
                
#                 throttle, brake = self.pid.compute_throttle_brake(self.target_speed, speed)
#                 steer, throttle, brake = self.safety.filter_control(steer, throttle, brake, speed)
                
#                 self.client.set_control(steer, throttle, brake)
                
#                 self.log_data.append({
#                     "time": current_time,
#                     "x": x,
#                     "y": y,
#                     "speed": speed,
#                     "target_speed": self.target_speed,
#                     "steer": steer,
#                     "throttle": throttle,
#                     "brake": brake,
#                     "yaw": yaw,
#                     "frame_idx": self.current_idx
#                 })
                
#                 if step_count % 50 == 0:
#                     print(f"Step {step_count}, speed={speed:.2f}, target={self.target_speed:.2f}, steer={steer:.3f}")
                
#                 self.client.step()
#                 step_count += 1
                
#         except KeyboardInterrupt:
#             print("\n⚠️ 用户中断")
#         except Exception as e:
#             print(f"❌ 运行时错误: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             if self.log_data and log_path:
#                 with open(log_path, 'w', newline='') as f:
#                     writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
#                     writer.writeheader()
#                     writer.writerows(self.log_data)
#                 print(f"✅ 日志已保存: {log_path}")
            
#             self.client.set_control(0, 0, 1)
#             time.sleep(0.5)
#             self.client.destroy()
#             print("仿真结束")

#     def _load_frame(self, idx):
#         """加载指定帧的轨迹"""
#         sample = self.samples[idx]
#         pred_traj = sample.get("parsed_trajectory")
#         if pred_traj and len(pred_traj) >= 6:
#             self.current_traj = [{"x": p[0], "y": p[1]} for p in pred_traj]
#             # 计算目标速度
#             total_dist = 0.0
#             for i in range(1, len(pred_traj)):
#                 dx = pred_traj[i][0] - pred_traj[i-1][0]
#                 dy = pred_traj[i][1] - pred_traj[i-1][1]
#                 total_dist += math.hypot(dx, dy)
#             self.target_speed = total_dist / (0.5 * (len(pred_traj)-1))
#             self.target_speed = max(2.0, min(8.0, self.target_speed))
#             print(f"帧 {idx+1}/{self.total}, 速度目标 {self.target_speed:.2f}")
#         else:
#             print(f"⚠️ 帧 {idx+1} 轨迹无效")
        
#         # 如果这是最后一帧，标记所有帧已加载
#         if idx == self.total - 1:
#             self.all_frames_loaded = True


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="运行预测轨迹仿真")
#     parser.add_argument("--json", type=str, required=True,
#                         help="predictions.json 文件路径")
#     parser.add_argument("--duration", type=float, default=None,
#                         help="仿真时长 (秒)，不指定则自动根据数据计算")
#     parser.add_argument("--start_speed", type=float, default=0.0,
#                         help="初始速度 (m/s)")
#     parser.add_argument("--controller", type=str, default="pure_pursuit",
#                         choices=["pure_pursuit", "mpc"],
#                         help="横向控制器类型: pure_pursuit 或 mpc")
#     args = parser.parse_args()
    
#     controller = PredictionReplayController(args.json, controller_type=args.controller)
#     controller.run(duration=args.duration, start_speed=args.start_speed)
#!/usr/bin/env python
"""
使用 OpenDriveVLA 预测轨迹在 CARLA 中仿真
支持两种横向控制器：
  - pure_pursuit: 纯跟踪 (基线)
  - mpc: 模型预测控制 (优化)

关键修正：
  1. 坐标映射：模型输出 (x=向右, y=向前) → 控制器期望 (x=向前, y=向左)
     映射规则：controller_x = model_y, controller_y = -model_x
  2. 车辆初始朝向设为 y 轴正方向 (yaw=90°)，以匹配预测轨迹的前进方向
  3. 基于 sim_t 时间戳精确加载帧
  4. 帧结束后保持跟踪 3 秒再停车
用法：
    python run_prediction_in_carla.py --json /path/to/predictions.json [--controller mpc]
    # --duration 参数已弃用，现在自动从数据中计算
"""
import sys
import os
import json
import time
import argparse
import math
import csv

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import carla

from control_module.core.trajectory_follower import PurePursuitController
from control_module.core.pid_controller import PIDController
from control_module.safety.safety_monitor import SafetyMonitor
from control_module.carla_integration.carla_client import CarlaClient
from control_module.core.mpc_controller import MPCController


class PredictionReplayController:
    def __init__(self, json_path, controller_type="pure_pursuit"):
        """
        初始化控制器
        
        Args:
            json_path: predictions.json 文件路径
            controller_type: "pure_pursuit" 或 "mpc"
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.samples = self.data.get("samples", [])
        self.total = len(self.samples)
        print(f"加载 {self.total} 帧预测数据")
        
        # ---- 计算数据总时长（用于自动确定仿真时间） ----
        if self.total > 0:
            first_sim_t = self.samples[0].get("sim_t", 0.0)
            last_sim_t = self.samples[-1].get("sim_t", 0.0)
            # 每帧预测覆盖未来 3 秒（最后一个点的 dt 约为 3.0）
            self.total_data_duration = (last_sim_t - first_sim_t) + 3.0
            # 加一点余量
            self.total_data_duration = max(5.0, self.total_data_duration + 1.0)
            print(f"自动计算数据总时长: {self.total_data_duration:.2f} 秒")
        else:
            self.total_data_duration = 30.0
        
        # ---- 记录第一帧的时间（用于帧更新计算） ----
        self.first_sim_t = self.samples[0].get("sim_t", 0.0) if self.total > 0 else 0.0
        self.last_sim_t = self.samples[-1].get("sim_t", 0.0) if self.total > 0 else 0.0
        
        # ---- 初始化控制器 ----
        self.controller_type = controller_type
        if controller_type == "mpc":
            self.mpc = MPCController(wheelbase=2.5, dt=0.1, N=15)
            print("✅ 使用 MPC 横向控制器")
        else:
            self.pure_pursuit = PurePursuitController(wheelbase=2.5, lookahead_time=0.8)
            print("✅ 使用 Pure Pursuit 横向控制器")
        
        # 纵向控制 (PID)
        self.pid = PIDController(kp=2.0, ki=0.15, kd=0.1)
        self.safety = SafetyMonitor()
        
        # CARLA 客户端
        self.client = CarlaClient()
        
        # 日志数据
        self.log_data = []
        
        # 当前目标轨迹（自车坐标系，已映射到控制器坐标系）
        self.current_traj = None
        self.target_speed = 3.0  # 默认目标速度
        
        # 帧更新状态
        self.current_idx = 0
        self.all_frames_loaded = False
        self.last_frame_load_time = 0.0
        self.frame_update_time = 0.0

    def _map_trajectory(self, pred_traj):
        """
        将模型输出的轨迹点映射到控制器坐标系。
        模型: x=向右, y=向前
        控制器: x=向前, y=向左
        映射: controller_x = model_y, controller_y = -model_x
        """
        mapped = []
        for p in pred_traj:
            # p = [model_x, model_y]
            mapped.append({"x": p[1], "y": -p[0]})
        return mapped

    def _load_frame(self, idx):
        """加载指定帧的轨迹并更新目标速度"""
        sample = self.samples[idx]
        pred_traj = sample.get("parsed_trajectory")
        if pred_traj and len(pred_traj) >= 6:
            self.current_traj = self._map_trajectory(pred_traj)
            # 从映射后的轨迹估算目标速度
            total_dist = 0.0
            for i in range(1, len(self.current_traj)):
                dx = self.current_traj[i]["x"] - self.current_traj[i-1]["x"]
                dy = self.current_traj[i]["y"] - self.current_traj[i-1]["y"]
                total_dist += math.hypot(dx, dy)
            self.target_speed = total_dist / (0.5 * (len(self.current_traj)-1))
            self.target_speed = max(2.0, min(8.0, self.target_speed))
            print(f"帧 {idx+1}/{self.total}, 速度目标 {self.target_speed:.2f}")
        else:
            # 无效轨迹，保持上一帧
            pass

    def run(self, start_speed=0.0):
        """主控制循环"""
        # 连接 CARLA
        if not self.client.connect():
            print("❌ 无法连接 CARLA，请检查服务器是否运行")
            return
        
        # 生成车辆：朝向 y 轴正方向 (yaw=90°)
        init_transform = carla.Transform(
            carla.Location(x=0, y=0, z=0.2),
            carla.Rotation(yaw=90)          # 车辆朝 y 正方向
        )
        if not self.client.spawn_vehicle(spawn_point=init_transform):
            print("❌ 车辆生成失败")
            return
        
        # 设置初始速度
        if start_speed > 0.1:
            vehicle = self.client.vehicle
            transform = vehicle.get_transform()
            local_vel = carla.Vector3D(x=0, y=start_speed, z=0)  # 因为车头朝 y 方向
            world_vel = transform.transform(local_vel)
            vehicle.set_target_velocity(world_vel)
        
        # 日志文件
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"prediction_replay_{timestamp}.csv")
        print(f"日志将保存至 {log_path}")
        
        # 主循环变量
        start_time = time.time()
        step_count = 0
        current_time = 0.0
        
        # 加载第一帧
        if self.total > 0:
            self._load_frame(0)
            self.current_idx = 1
            self.last_frame_load_time = 0.0
            self.frame_update_time = 0.0
        
        try:
            while current_time < self.total_data_duration:
                current_time = time.time() - start_time
                
                # ---- 获取车辆状态 ----
                x, y, yaw, speed = self.client.get_vehicle_state()
                vehicle_transform = self.client.vehicle.get_transform()
                ego_x = vehicle_transform.location.x
                ego_y = vehicle_transform.location.y
                ego_yaw = math.radians(vehicle_transform.rotation.yaw)
                
                # ---- 帧更新逻辑 ----
                if not self.all_frames_loaded and self.current_idx < self.total:
                    # 根据 sim_t 时间戳加载帧
                    next_sample = self.samples[self.current_idx]
                    next_sim_t = next_sample.get("sim_t", 0.0)
                    # 计算相对于第一帧的期望时间
                    expected_time = (next_sim_t - self.first_sim_t)
                    # 如果当前时间已经超过期望时间，加载下一帧
                    if current_time >= expected_time:
                        self._load_frame(self.current_idx)
                        self.current_idx += 1
                        self.last_frame_load_time = current_time
                        self.frame_update_time = current_time
                
                # ---- 检查是否所有帧已加载完 ----
                if self.current_idx >= self.total:
                    self.all_frames_loaded = True
                    # 如果所有帧已加载完，检查是否超过最后一帧预测范围（3秒）
                    if self.last_frame_load_time > 0 and (current_time - self.last_frame_load_time) > 3.0:
                        # 超出预测范围，清空轨迹并停车
                        self.current_traj = None
                        self.target_speed = 0.0
                
                # ---- 控制计算 ----
                if self.current_traj is not None:
                    # 横向控制
                    if self.controller_type == "mpc":
                        steer = self.mpc.compute_steer(
                            self.current_traj,
                            speed,
                            ego_x,
                            ego_y,
                            ego_yaw
                        )
                    else:
                        steer, _ = self.pure_pursuit.compute_steer(self.current_traj, speed)
                else:
                    steer = 0.0
                
                # 纵向控制
                throttle, brake = self.pid.compute_throttle_brake(self.target_speed, speed)
                
                # 安全过滤
                steer, throttle, brake = self.safety.filter_control(steer, throttle, brake, speed)
                
                # 发送控制指令
                self.client.set_control(steer, throttle, brake)
                
                # ---- 记录日志 ----
                self.log_data.append({
                    "time": current_time,
                    "x": x,
                    "y": y,
                    "speed": speed,
                    "target_speed": self.target_speed,
                    "steer": steer,
                    "throttle": throttle,
                    "brake": brake,
                    "yaw": yaw,
                    "frame_idx": self.current_idx
                })
                
                # ---- 打印状态 ----
                if step_count % 50 == 0:
                    print(f"Step {step_count}, speed={speed:.2f}, target={self.target_speed:.2f}, steer={steer:.3f}")
                
                # 步进
                self.client.step()
                step_count += 1
                
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # ---- 保存日志 ----
            if self.log_data and log_path:
                with open(log_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.log_data)
                print(f"✅ 日志已保存: {log_path}")
            
            # 停止车辆并清理
            self.client.set_control(0, 0, 1)  # 刹车
            time.sleep(0.5)
            self.client.destroy()
            print("仿真结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行预测轨迹仿真")
    parser.add_argument("--json", type=str, required=True,
                        help="predictions.json 文件路径")
    parser.add_argument("--start_speed", type=float, default=0.0,
                        help="初始速度 (m/s)")
    parser.add_argument("--controller", type=str, default="pure_pursuit",
                        choices=["pure_pursuit", "mpc"],
                        help="横向控制器类型: pure_pursuit 或 mpc")
    args = parser.parse_args()
    
    controller = PredictionReplayController(args.json, controller_type=args.controller)
    controller.run(start_speed=args.start_speed)