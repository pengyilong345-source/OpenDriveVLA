"""
CARLA 客户端封装
支持生成车辆在指定位置，获取状态，发送控制。
"""
import carla
import time
import math
from typing import Optional, Tuple


class CarlaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.vehicle = None
        self.ego_vehicle = None
        self.control = carla.VehicleControl()
        self.dt = 0.05

    def connect(self) -> bool:
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
        生成车辆，如果 spawn_point 为 None，则使用地图第一个生成点。
        """
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(blueprint_name)

            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = spawn_points[0] if spawn_points else carla.Transform()

            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.ego_vehicle = self.vehicle
            print(f"✅ 车辆已生成: {blueprint_name}")
            print(f"   位置: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})")
            return True
        except Exception as e:
            print(f"❌ 车辆生成失败: {e}")
            return False

    def get_vehicle_state(self) -> Tuple[float, float, float, float]:
        """返回 (x, y, yaw_rad, speed_mps)"""
        if self.vehicle is None:
            return 0.0, 0.0, 0.0, 0.0
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return x, y, yaw, speed

    def get_transform(self) -> carla.Transform:
        """返回当前车辆的 Transform"""
        if self.vehicle is None:
            return carla.Transform()
        return self.vehicle.get_transform()

    def set_control(self, steer: float, throttle: float, brake: float):
        if self.vehicle is None:
            return
        self.control.steer = float(steer)
        self.control.throttle = float(throttle)
        self.control.brake = float(brake)
        self.vehicle.apply_control(self.control)

    def step(self):
        self.world.tick()

    def destroy(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
            print("车辆已销毁")