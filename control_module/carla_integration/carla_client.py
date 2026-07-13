"""
CARLA 客户端封装
负责：连接CARLA服务器、生成车辆、获取传感器数据、发送控制指令
"""

import carla
import time
import math
from typing import Optional, Tuple


class CarlaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 10.0):
        """
        初始化 CARLA 客户端。
        
        Args:
            host: CARLA 服务器IP
            port: CARLA 服务器端口
            timeout: 连接超时时间（秒）
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.vehicle = None
        self.ego_vehicle = None
        
        # 控制参数
        self.control = carla.VehicleControl()
        self.dt = 0.05  # 控制周期 (50Hz)
        
    def connect(self) -> bool:
        """连接CARLA服务器"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            print(f"✅ 已连接到CARLA服务器 (版本: {self.client.get_client_version()})")
            print(f"   地图: {self.world.get_map().name}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def spawn_vehicle(self, blueprint_name: str = "vehicle.audi.tt", 
                      spawn_point: Optional[carla.Transform] = None) -> bool:
        """
        在CARLA世界中生成一辆车。
        
        Args:
            blueprint_name: 车辆蓝图名称 (如 'vehicle.audi.tt', 'vehicle.mercedes.coupe')
            spawn_point: 生成位置 (可选)
        """
        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(blueprint_name)
            
            # 如果没有指定生成点，使用地图的随机点
            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = spawn_points[0] if spawn_points else carla.Transform()
            
            # 生成车辆
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.ego_vehicle = self.vehicle
            print(f"✅ 车辆已生成: {blueprint_name}")
            print(f"   位置: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})")
            return True
            
        except Exception as e:
            print(f"❌ 车辆生成失败: {e}")
            return False
    
    def get_vehicle_state(self) -> Tuple[float, float, float, float]:
        """
        获取车辆当前状态。
        
        Returns:
            (x, y, yaw, speed_mps): 位置(x,y)，航向角(rad)，速度(m/s)
        """
        if self.vehicle is None:
            return 0.0, 0.0, 0.0, 0.0
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)  # CARLA用度，转换为弧度
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
        
        return x, y, yaw, speed
    
    def set_control(self, steer: float, throttle: float, brake: float):
        """
        发送控制指令给车辆。
        
        Args:
            steer: 转向角 [-1.0, 1.0]
            throttle: 油门 [0.0, 1.0]
            brake: 刹车 [0.0, 1.0]
        """
        if self.vehicle is None:
            return
        
        self.control.steer = float(steer)
        self.control.throttle = float(throttle)
        self.control.brake = float(brake)
        self.vehicle.apply_control(self.control)
    
    def step(self):
        """推进一帧 (CARLA tick)"""
        self.world.tick()
    
    def destroy(self):
        """销毁车辆，关闭连接"""
        if self.vehicle is not None:
            self.vehicle.destroy()
            print("车辆已销毁")
        if self.client is not None:
            # 不关闭client，因为它是轻量级连接
            pass


# --- 简单测试入口 ---
if __name__ == "__main__":
    print("测试CARLA客户端连接...")
    print("请确保CARLA服务器正在运行 (CarlaUE4.exe)")
    print("-" * 50)
    
    client = CarlaClient()
    if not client.connect():
        print("请启动CARLA服务器后再运行此脚本")
        exit(1)
    
    if not client.spawn_vehicle():
        print("车辆生成失败")
        exit(1)
    
    print("正在测试控制指令... (5秒后停止)")
    for i in range(50):  # 5秒 (50 * 0.1s)
        # 简单测试：原地转向
        steer = 0.5 * math.sin(i * 0.1)
        throttle = 0.3
        brake = 0.0
        client.set_control(steer, throttle, brake)
        client.step()
        time.sleep(0.05)  # 模拟控制周期
    
    print("测试完成，车辆停止")
    client.set_control(0.0, 0.0, 0.0)  # 停车
    time.sleep(1)
    
    client.destroy()