"""
横向误差分析工具
功能：计算实际路径相对于参考轨迹的横向偏差（法向距离），而非简单的欧几里得距离。
用法：
    默认：python compare_lateral_error.py
    指定日志：python compare_lateral_error.py logs/xxx.csv
"""
import os
import sys
import glob
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_DATA_DIR = r"E:\course_file\OpenDriveVLA\our_work\OpenDriveVLA\control_module\HighwayExit_Town06_Route291_Weather5\annotations"
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "output")


def find_latest_log(log_dir='../logs'):
    log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
    if not files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
    return max(files, key=os.path.getmtime)


def load_frames(data_dir):
    import json
    import glob
    files = glob.glob(os.path.join(data_dir, "frame_*.json"))
    
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
    world_points = []
    for pt in traj_points:
        dx = pt['x']
        dy = -pt['y']   # 取反，与 run_control_loop 保持一致
        wx = ego_x + dx * math.cos(ego_yaw) - dy * math.sin(ego_yaw)
        wy = ego_y + dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
        world_points.append((wx, wy))
    return world_points


def compute_lateral_errors(actual_x, actual_y, ref_points):
    """
    计算实际点的横向误差（法向距离）。
    对于每个实际点，找到参考轨迹上最近的两个点，计算法向量投影。
    """
    lateral_errors = []
    nearest_indices = []
    ref_x_all = ref_points[:, 0]
    ref_y_all = ref_points[:, 1]

    for ax, ay in zip(actual_x, actual_y):
        dists = np.hypot(ax - ref_x_all, ay - ref_y_all)
        idx = np.argmin(dists)
        nearest_indices.append(idx)

        # 获取最近点前后的点以计算切线方向
        idx_prev = max(0, idx - 1)
        idx_next = min(len(ref_x_all) - 1, idx + 1)
        
        dx = ref_x_all[idx_next] - ref_x_all[idx_prev]
        dy = ref_y_all[idx_next] - ref_y_all[idx_prev]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            lateral_errors.append(0.0)
            continue
        
        # 单位切向量 (Tx, Ty)
        Tx, Ty = dx / norm, dy / norm
        # 法向量 (Nx, Ny) 指向左侧
        Nx, Ny = -Ty, Tx
        
        # 实际点相对最近参考点的偏移
        ex = ax - ref_x_all[idx]
        ey = ay - ref_y_all[idx]
        
        # 横向误差 = 偏移在法向量上的投影
        lateral_error = ex * Nx + ey * Ny
        lateral_errors.append(lateral_error)
    
    return np.array(lateral_errors), np.array(nearest_indices)


def main():
    parser = argparse.ArgumentParser(description="横向误差分析")
    parser.add_argument("log_path", nargs='?', help="日志文件路径（可选）")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"数据集目录（默认: {DEFAULT_DATA_DIR}）")
    parser.add_argument("--save_to_dir", "-s", type=str, default=DEFAULT_SAVE_DIR,
                        help=f"保存图片的目录（默认: {DEFAULT_SAVE_DIR}）")
    args = parser.parse_args()

    if args.log_path:
        log_path = args.log_path
    else:
        try:
            log_path = find_latest_log()
            print(f"自动选择最新日志: {log_path}")
        except FileNotFoundError as e:
            print(e)
            return

    df = pd.read_csv(log_path)
    print(f"✅ 加载日志: {log_path}，共 {len(df)} 帧")
    
    if 'x' not in df.columns or 'y' not in df.columns:
        print("⚠️ 日志缺少 x 或 y 字段")
        return
    
    if 'time' not in df.columns:
        df['time'] = df.index * 0.1

    actual_x = df['x'].values
    actual_y = df['y'].values
    times = df['time'].values
    speed = df['speed'].values if 'speed' in df.columns else None
    
    print(f"加载数据集: {args.data_dir}")
    frames = load_frames(args.data_dir)
    print(f"✅ 加载了 {len(frames)} 帧")
    
    # 提取所有参考轨迹点
    ref_x_all, ref_y_all = [], []
    for frame in frames:
        ego = frame["ego_state"]
        traj = frame.get("future_trajectory_ego_frame", [])
        if traj:
            ex, ey, eyaw = ego["x"], ego["y"], ego["yaw"]
            world_pts = traj_to_world(ex, ey, eyaw, traj)
            for wx, wy in world_pts:
                ref_x_all.append(wx)
                ref_y_all.append(wy)
    
    if not ref_x_all:
        print("⚠️ 未提取到参考轨迹点")
        return
    
    ref_pts = np.array(list(zip(ref_x_all, ref_y_all)))
    print(f"✅ 提取了 {len(ref_pts)} 个参考轨迹点")

    # 计算横向误差
    lateral_errors, nearest_indices = compute_lateral_errors(actual_x, actual_y, ref_pts)
    
    # 统计
    mean_abs_lat = np.mean(np.abs(lateral_errors))
    max_abs_lat = np.max(np.abs(lateral_errors))
    std_lat = np.std(lateral_errors)
    # 超过 0.5m（半车道宽）的比例
    exceed_ratio = np.sum(np.abs(lateral_errors) > 0.5) / len(lateral_errors) * 100

    print(f"\n--- 横向误差统计 ---")
    print(f"平均绝对横向误差: {mean_abs_lat:.3f} m")
    print(f"最大绝对横向误差: {max_abs_lat:.3f} m")
    print(f"标准差: {std_lat:.3f} m")
    print(f"误差 > 0.5m 的比例: {exceed_ratio:.2f}%")

    # ===== 绘图 =====
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('横向误差分析', fontsize=16)

    # 图1：横向误差随时间变化
    ax = axes[0]
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=0.8, label='车道边界 (0.5m)')
    ax.axhline(-0.5, color='orange', linestyle='--', linewidth=0.8)
    ax.plot(times, lateral_errors, 'b-', linewidth=1.0, alpha=0.7, label='横向误差')
    ax.fill_between(times, 0.5, -0.5, color='green', alpha=0.1)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('横向误差 (m)')
    ax.set_title(f'横向误差 | 均值={mean_abs_lat:.3f}, 最大={max_abs_lat:.3f}, >0.5m={exceed_ratio:.1f}%')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图2：速度 vs 横向误差（散点）
    ax = axes[1]
    if speed is not None:
        ax.scatter(speed, lateral_errors, c='blue', s=5, alpha=0.5)
        ax.set_xlabel('车速 (m/s)')
        ax.set_ylabel('横向误差 (m)')
        ax.set_title('速度 vs 横向误差')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, '日志缺少 speed 字段', ha='center', va='center')

    # 图3：路径对比（标注误差较大区域）
    ax = axes[2]
    ax.plot(ref_x_all, ref_y_all, 'g-', linewidth=1.0, alpha=0.3, label='参考轨迹')
    ax.plot(actual_x, actual_y, 'b-', linewidth=1.5, alpha=0.8, label='实际路径')
    # 标记误差最大的10个点
    top10_idx = np.argsort(np.abs(lateral_errors))[-10:]
    ax.scatter(actual_x[top10_idx], actual_y[top10_idx], c='red', s=30, label='误差最大点')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('路径对比（红点: 横向误差最大处）')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    # 保存
    save_dir = os.path.join(args.save_to_dir, 'lateral')
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ 图片已保存: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()