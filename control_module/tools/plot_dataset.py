"""
数据集坐标可视化（Python + matplotlib）
功能：
    - 绘制所有帧的 ego_state 位置（红色路径）
    - 绘制所有帧的 future_trajectory_ego_frame 转换后的世界坐标（绿色散点）
    - 绘制前几帧的轨迹线（便于观察方向）
"""

import os
import json
import math
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_frames(data_dir):
    """加载所有 frame_*.json 并排序"""
    pattern = os.path.join(data_dir, "frame_*.json")
    files = glob.glob(pattern)
    
    def extract_frame_id(path):
        base = os.path.basename(path)
        num_str = base.split('_')[1].split('.')[0]
        return int(num_str)
    
    files.sort(key=extract_frame_id)
    
    frames = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            frames.append(json.load(fp))
    return frames


def traj_to_world(ego_x, ego_y, ego_yaw, traj_points):
    """
    将自车坐标系下的轨迹点转换到世界坐标系
    traj_points: 列表，每个元素包含 x, y
    """
    world_points = []
    for pt in traj_points:
        dx = pt['x']
        dy = -pt['y'] # 取负值以适应坐标系转换，改过后绿点和红线重合度极佳
        wx = ego_x + dx * math.cos(ego_yaw) - dy * math.sin(ego_yaw)
        wy = ego_y + dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
        world_points.append((wx, wy))
    return world_points


def main():
    # ====== 配置 ======
    data_dir = r"E:\course_file\OpenDriveVLA\our_work\OpenDriveVLA\control_module\HighwayExit_Town06_Route291_Weather5\annotations"
    # 请将上述路径改为你自己的数据目录
    
    # ====== 加载数据 ======
    print("加载数据...")
    frames = load_frames(data_dir)
    print(f"✅ 加载了 {len(frames)} 帧")
    
    # ====== 提取数据 ======
    ego_x, ego_y = [], []
    traj_x, traj_y = [], []
    trajectory_lines = []  # 用于画前几帧的轨迹线
    
    num_frames_to_show_lines = 30  # 显示前30帧的轨迹连线
    
    for idx, frame in enumerate(frames):
        ego = frame['ego_state']
        ex, ey = ego['x'], ego['y']
        yaw = ego['yaw']
        
        ego_x.append(ex)
        ego_y.append(ey)
        
        # 转换轨迹点
        traj = frame.get('future_trajectory_ego_frame', [])
        if traj:
            world_pts = traj_to_world(ex, ey, yaw, traj)
            for wx, wy in world_pts:
                traj_x.append(wx)
                traj_y.append(wy)
            
            # 记录前几帧的轨迹线
            if idx < num_frames_to_show_lines:
                trajectory_lines.append((ex, ey, world_pts))
    
    # ====== 绘图 ======
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 1. 绘制所有 ego_state 路径（红色连线）
    ax.plot(ego_x, ego_y, 'r-', linewidth=1.5, alpha=0.7, label='Ego Path (all frames)')
    ax.scatter(ego_x[0], ego_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(ego_x[-1], ego_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    
    # 2. 绘制所有轨迹点（绿色散点）
    ax.scatter(traj_x, traj_y, c='lime', s=8, alpha=0.5, label='Future Trajectory Points', zorder=1)
    
    # 3. 绘制前几帧的轨迹连线（从 ego 位置延伸出去）
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_lines)))
    for i, (ex, ey, pts) in enumerate(trajectory_lines):
        if len(pts) >= 2:
            # 从 ego 位置到第一个轨迹点的连线
            xs = [ex] + [p[0] for p in pts]
            ys = [ey] + [p[1] for p in pts]
            ax.plot(xs, ys, color=colors[i], linewidth=1.0, alpha=0.6)
    
    # 4. 图形美化（所有显示文字改为英文）
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Dataset Coordinate Visualization\nTotal {len(frames)} frames, {len(traj_x)} trajectory points', fontsize=14)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # ====== 输出统计信息（英文） ======
    print("\n--- Statistics ---")
    print(f"Total frames: {len(frames)}")
    print(f"Ego points: {len(ego_x)}")
    print(f"Trajectory points: {len(traj_x)}")
    print(f"X range: [{min(ego_x):.2f}, {max(ego_x):.2f}]")
    print(f"Y range: [{min(ego_y):.2f}, {max(ego_y):.2f}]")
    
    # 检查跳变
    dx = np.diff(ego_x)
    dy = np.diff(ego_y)
    jumps = np.sqrt(dx**2 + dy**2)
    if len(jumps) > 0:
        print(f"Max frame-to-frame jump: {max(jumps):.2f} m")
        if max(jumps) > 5:
            print("⚠️ Warning: Large jump detected (>5m), possible data discontinuity")
        else:
            print("✅ Frame-to-frame jumps are normal")
    
    # ====== 显示图形 ======
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()