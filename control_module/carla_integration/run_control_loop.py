"""
CARLA 控制主循环：将控制模块挂载到CARLA仿真器中
支持在仿真中绘制目标轨迹点，便于肉眼验证
"""

import sys
import os
import time
import csv
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


class CarlaControlLoop:
    def __init__(self, trajectory_file: str = None):
        """
        初始化CARLA控制循环。
        
        Args:
            trajectory_file: 轨迹JSON文件路径 (如 mock_trajectories.json)
        """
        self.trajectory_file = trajectory_file
        self.trajectory = None
        self.dense_trajectory = None   # 用于绘制的密集点
        self.target_speed = 3.0
        
        # 加载轨迹
        if trajectory_file and os.path.exists(trajectory_file):
            with open(trajectory_file, "r") as f:
                data = json.load(f)
                self.trajectory = data["future_trajectory_ego_frame"]
                self.target_speed = data["ego_state"]["speed"]
            print(f"✅ 已加载轨迹: {len(self.trajectory)} 个点")
        else:
            print("⚠️ 未提供轨迹文件，使用默认直线轨迹")
            self.trajectory = self._generate_default_trajectory()
        
        # 生成密集轨迹供可视化
        self.dense_trajectory = interpolate_trajectory(self.trajectory, step=0.2)
        print(f"密集轨迹点数: {len(self.dense_trajectory)} (用于显示)")
        
        # 初始化控制器
        self.pid = PIDController(kp=0.6, ki=0.12, kd=0.08)
        self.pure_pursuit = PurePursuitController(wheelbase=2.8, lookahead_time=1.2)
        self.safety = SafetyMonitor()
        
        # CARLA客户端
        self.client = CarlaClient()
        
        # 日志记录
        self.log_data = []
        self.log_file = None
        
        # 调试绘图标志
        self.draw_enabled = True
    
    def _generate_default_trajectory(self) -> List[Dict]:
        """生成默认的直线轨迹（用于测试）"""
        points = []
        for i in range(1, 7):
            dt = i * 0.5
            points.append({
                "dt": round(dt, 2),
                "x": round(3.0 * dt, 2),
                "y": 0.0
            })
        return points
    
    def draw_trajectory(self, vehicle_transform: carla.Transform, 
                        trajectory_points: List[Dict], 
                        color: carla.Color = carla.Color(0, 255, 0),
                        size: float = 0.2,
                        life_time: float = 0.1) -> None:
        """
        在CARLA世界中绘制轨迹点（绿色点）
        
        Args:
            vehicle_transform: 当前车辆的变换（用于将相对坐标转为世界坐标）
            trajectory_points: 轨迹点列表，每个点包含 'x', 'y'
            color: 点颜色
            size: 点大小
            life_time: 显示时长（秒），设为0.1则每帧刷新
        """
        if not self.draw_enabled:
            return
        world = self.client.world
        debug = world.debug
        
        for pt in trajectory_points:
            # 构造相对位置（z=0.2 抬高一点，避免与地面重叠）
            local_location = carla.Location(x=pt['x'], y=pt['y'], z=0.2)
            # 转换为世界坐标
            world_location = vehicle_transform.transform(local_location)
            debug.draw_point(
                world_location,
                size=size,
                color=color,
                life_time=life_time
            )
    
    def draw_trajectory_with_labels(self, vehicle_transform: carla.Transform, 
                                 trajectory_points: List[Dict], 
                                 color: carla.Color = carla.Color(0, 255, 0),
                                 size: float = 0.3,
                                 life_time: float = 0.1) -> None:
        """
        绘制原始6个目标点，并在上方显示数字标签 [1] ~ [6]
        """
        if not self.draw_enabled:
            return
        world = self.client.world
        debug = world.debug
        
        for idx, pt in enumerate(trajectory_points):
            # 1. 绘制点（球体）
            local_location = carla.Location(x=pt['x'], y=pt['y'], z=0.1)
            world_location = vehicle_transform.transform(local_location)
            debug.draw_point(
                world_location,
                size=size,
                color=color,
                life_time=life_time
            )
            
            # 2. 绘制数字标签（抬高一点，避免被遮挡）
            text_location = carla.Location(x=pt['x'], y=pt['y'], z=1.2)  # 抬高1.2米
            text_world_location = vehicle_transform.transform(text_location)
            debug.draw_string(
                text_world_location,
                f'  [{idx+1}]',  # 显示 [1] [2] ... [6]
                draw_shadow=False,
                color=carla.Color(255, 255, 255),  # 白色
                life_time=life_time
            )

    def run(self, duration_seconds: float = 30.0):
        """
        运行CARLA控制循环。
        
        Args:
            duration_seconds: 运行持续时间（秒）
        """
        print("\n" + "=" * 60)
        print("🚗 CARLA 控制循环启动")
        print("=" * 60)
        
        # 1. 连接CARLA
        if not self.client.connect():
            print("❌ 无法连接到CARLA，请检查服务器是否运行")
            return
        
        # 2. 生成车辆
        if not self.client.spawn_vehicle():
            print("❌ 车辆生成失败")
            return
        
        # 3. 初始化日志文件
        import time
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"carla_control_{timestamp}.csv")
        
        print(f"日志将保存到: {self.log_file}")
        print("-" * 60)
        print("时间(s) | 车速(m/s) | 油门 | 刹车 | 转向 | 横向误差")
        print("-" * 60)
        
        # 4. 主控制循环
        start_time = time.time()
        last_print_time = start_time
        step_count = 0
        
        try:
            while True:
                current_time = time.time() - start_time
                if current_time > duration_seconds:
                    break
                
                # 获取车辆状态
                x, y, yaw, speed = self.client.get_vehicle_state()
                vehicle_transform = self.client.vehicle.get_transform()
                
                # 计算控制量
                steer, lookahead = self.pure_pursuit.compute_steer(
                    self.trajectory, speed
                )
                throttle, brake = self.pid.compute_throttle_brake(
                    self.target_speed, speed
                )
                
                # 安全过滤
                steer, throttle, brake = self.safety.filter_control(
                    steer, throttle, brake, speed
                )
                
                # 发送控制指令
                self.client.set_control(steer, throttle, brake)
                
                # ===== 可视化轨迹点 =====
                # 绘制原始6个目标点（绿色）
                # self.draw_trajectory(
                #     vehicle_transform, 
                #     self.trajectory, 
                #     color=carla.Color(0, 255, 0),  # 绿色
                #     size=0.25,
                #     life_time=0.1
                # )

                # 替换原来绘制绿色点的代码
                # self.draw_trajectory_with_labels(
                #     vehicle_transform, 
                #     self.trajectory, 
                #     color=carla.Color(0, 255, 0),  # 绿色
                #     size=0.3,
                #     life_time=0.1
                # )

                # # 绘制密集插值路径（黄色小点）
                # self.draw_trajectory(
                #     vehicle_transform,
                #     self.dense_trajectory,
                #     color=carla.Color(255, 255, 0),  # 黄色
                #     size=0.1,
                #     life_time=0.1
                # )
                # # 绘制当前前瞻点（红色大点）
                # if lookahead is not None:
                #     debug = self.client.world.debug
                #     lookahead_local = carla.Location(x=lookahead[0], y=lookahead[1], z=0.5)
                #     lookahead_world = vehicle_transform.transform(lookahead_local)
                #     debug.draw_point(
                #         lookahead_world,
                #         size=0.3,
                #         color=carla.Color(255, 0, 0),  # 红色
                #         life_time=0.1
                #     )
                
                # 记录日志
                lateral_error = abs(y)  # 近似横向误差（假设目标在y=0附近）
                self.log_data.append({
                    "time": current_time,
                    "x": x,
                    "y": y,
                    "speed": speed,
                    "steer": steer,
                    "throttle": throttle,
                    "brake": brake,
                    "yaw": yaw
                })
                
                # 打印状态 (每0.5秒)
                if current_time - last_print_time >= 0.5:
                    print(f"{current_time:6.1f}  | {speed:8.2f} | {throttle:5.3f} | {brake:5.3f} | {steer:6.3f} | {lateral_error:8.4f}")
                    last_print_time = current_time
                
                # 推进一帧
                self.client.step()
                step_count += 1
                
                # 安全检测：如果跑出地图边界，终止
                if abs(x) > 100 or abs(y) > 100:
                    print(f"⚠️ 车辆超出边界，终止运行")
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断，正在停止...")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理
            print("-" * 60)
            print("正在停止车辆并保存日志...")
            self.client.set_control(0.0, 0.0, 1.0)  # 刹车停车
            time.sleep(0.5)
            self.client.destroy()
            
            # 保存日志
            if self.log_data and self.log_file:
                import csv
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.log_data)
                print(f"✅ 日志已保存: {self.log_file}")
            
            print(f"总步数: {step_count}")
            print("=" * 60)


if __name__ == "__main__":
    # 使用默认轨迹文件或参数
    trajectory_file = None
    if len(sys.argv) > 1:
        trajectory_file = sys.argv[1]
    else:
        default_mock = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "test_data", "mock_trajectories.json"
        )
        if os.path.exists(default_mock):
            trajectory_file = default_mock
            print(f"使用默认轨迹文件: {trajectory_file}")
    
    loop = CarlaControlLoop(trajectory_file)
    loop.run(duration_seconds=20.0)