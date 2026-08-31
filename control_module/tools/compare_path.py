# """
# 路径对比分析工具
# 功能：对比实际行驶路径与数据集参考路径
# 用法：
#     默认：python compare_path.py
#     指定日志：python compare_path.py logs/xxx.csv
#     指定输出目录：python compare_path.py --save_to_dir ./my_analysis
# """
# import os
# import sys
# import glob
# import math
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# DEFAULT_DATA_DIR = r"E:\course_file\OpenDriveVLA\our_work\OpenDriveVLA\control_module\HighwayExit_Town06_Route291_Weather5\annotations"
# DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "output")


# def find_latest_log(log_dir='../logs'):
#     log_dir = os.path.join(os.path.dirname(__file__), log_dir)
#     files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
#     if not files:
#         raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
#     return max(files, key=os.path.getmtime)


# def load_frames(data_dir):
#     import json
#     import glob
#     files = glob.glob(os.path.join(data_dir, "frame_*.json"))
    
#     def extract_frame_id(path):
#         base = os.path.basename(path)
#         num_str = base.split('_')[1].split('.')[0]
#         return int(num_str)
    
#     files.sort(key=extract_frame_id)
#     frames = []
#     for f in files:
#         with open(f, 'r', encoding='utf-8') as fp:
#             frames.append(json.load(fp))
#     return frames


# def traj_to_world(ego_x, ego_y, ego_yaw, traj_points):
#     world_points = []
#     for pt in traj_points:
#         dx = pt['x']
#         dy = -pt['y']
#         wx = ego_x + dx * math.cos(ego_yaw) - dy * math.sin(ego_yaw)
#         wy = ego_y + dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
#         world_points.append((wx, wy))
#     return world_points


# def main():
#     parser = argparse.ArgumentParser(description="路径对比分析")
#     parser.add_argument("log_path", nargs='?', help="日志文件路径（可选，不指定则自动选最新）")
#     parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
#                         help=f"数据集目录（默认: {DEFAULT_DATA_DIR}）")
#     parser.add_argument("--save_to_dir", "-s", type=str, default=DEFAULT_SAVE_DIR,
#                         help=f"保存图片的目录（默认: {DEFAULT_SAVE_DIR}）")
#     args = parser.parse_args()

#     if args.log_path:
#         log_path = args.log_path
#     else:
#         try:
#             log_path = find_latest_log()
#             print(f"自动选择最新日志: {log_path}")
#         except FileNotFoundError as e:
#             print(e)
#             return

#     df = pd.read_csv(log_path)
#     print(f"✅ 加载日志: {log_path}，共 {len(df)} 帧")
    
#     if 'x' not in df.columns or 'y' not in df.columns:
#         print("⚠️ 日志缺少 x 或 y 字段")
#         return
    
#     actual_x = df['x'].values
#     actual_y = df['y'].values
    
#     print(f"加载数据集: {args.data_dir}")
#     frames = load_frames(args.data_dir)
#     print(f"✅ 加载了 {len(frames)} 帧")
    
#     ref_x_all, ref_y_all = [], []
#     for frame in frames:
#         ego = frame["ego_state"]
#         traj = frame.get("future_trajectory_ego_frame", [])
#         if traj:
#             ex, ey, eyaw = ego["x"], ego["y"], ego["yaw"]
#             world_pts = traj_to_world(ex, ey, eyaw, traj)
#             for wx, wy in world_pts:
#                 ref_x_all.append(wx)
#                 ref_y_all.append(wy)
    
#     print(f"✅ 提取了 {len(ref_x_all)} 个参考轨迹点")

#     fig, ax = plt.subplots(figsize=(14, 10))

#     ax.scatter(ref_x_all, ref_y_all, c='lime', s=12, alpha=0.5, label='参考轨迹点', zorder=1)
#     ax.plot(actual_x, actual_y, 'b-', linewidth=2.5, alpha=0.9, label='实际路径', zorder=2)
#     ax.scatter(actual_x[0], actual_y[0], c='blue', s=120, marker='o', edgecolors='white', label='实际起点', zorder=4)
#     ax.scatter(actual_x[-1], actual_y[-1], c='blue', s=120, marker='s', edgecolors='white', label='实际终点', zorder=4)

#     if frames:
#         first_ego = frames[0]["ego_state"]
#         last_ego = frames[-1]["ego_state"]
#         ax.scatter(first_ego["x"], first_ego["y"], c='green', s=180, marker='o', 
#                    edgecolors='black', linewidths=1.5, label='数据集起点', zorder=5)
#         ax.scatter(last_ego["x"], last_ego["y"], c='green', s=180, marker='s', 
#                    edgecolors='black', linewidths=1.5, label='数据集终点', zorder=5)

#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_title(f'实际路径 vs 参考轨迹\n日志: {os.path.basename(log_path)}')
#     ax.axis('equal')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     # 计算偏离距离
#     if ref_x_all and len(actual_x) > 0:
#         sample_idx = np.linspace(0, len(actual_x)-1, min(100, len(actual_x)), dtype=int)
#         ref_pts = np.array(list(zip(ref_x_all, ref_y_all)))
#         distances = []
#         for idx in sample_idx:
#             if idx < len(actual_x):
#                 dists = np.hypot(actual_x[idx] - ref_pts[:, 0], actual_y[idx] - ref_pts[:, 1])
#                 distances.append(np.min(dists))
#         if distances:
#             ax.text(0.02, 0.02, f'平均偏离: {np.mean(distances):.2f} m\n最大偏离: {np.max(distances):.2f} m',
#                     transform=ax.transAxes, fontsize=12,
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

#     plt.tight_layout()

#     save_dir = os.path.join(args.save_to_dir, 'path')
#     os.makedirs(save_dir, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(log_path))[0]
#     save_path = os.path.join(save_dir, f"{base_name}.png")
#     plt.savefig(save_path, dpi=150)
#     print(f"✅ 图片已保存: {save_path}")

#     plt.show()


# if __name__ == "__main__":
#     main()
"""
路径对比分析工具
功能：对比实际行驶路径与数据集参考路径（所有帧的目标轨迹点）
用法：
    默认：python compare_path.py
    指定日志：python compare_path.py logs/xxx.csv
    指定输出目录：python compare_path.py --save_to_dir ./my_analysis
输出：
    - 图片（路径对比图）
    - CSV 文件（实际 vs 期望坐标列表）
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


def main():
    parser = argparse.ArgumentParser(description="路径对比分析")
    parser.add_argument("log_path", nargs='?', help="日志文件路径（可选，不指定则自动选最新）")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"数据集目录（默认: {DEFAULT_DATA_DIR}）")
    parser.add_argument("--save_to_dir", "-s", type=str, default=DEFAULT_SAVE_DIR,
                        help=f"保存图片和CSV的目录（默认: {DEFAULT_SAVE_DIR}）")
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
        print("⚠️ 日志缺少 time 字段，将使用索引作为时间")
        df['time'] = df.index * 0.1  # 假设 0.1s 间隔

    actual_x = df['x'].values
    actual_y = df['y'].values
    times = df['time'].values
    
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

    # ===== 为每个实际点找到最近参考点 =====
    expected_x = []
    expected_y = []
    errors = []
    for ax, ay in zip(actual_x, actual_y):
        dists = np.hypot(ax - ref_pts[:, 0], ay - ref_pts[:, 1])
        idx = np.argmin(dists)
        ex, ey = ref_pts[idx, 0], ref_pts[idx, 1]
        expected_x.append(ex)
        expected_y.append(ey)
        errors.append(dists[idx])
    
    # ===== 绘图 =====
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(ref_x_all, ref_y_all, c='lime', s=12, alpha=0.5, label='参考轨迹点', zorder=1)
    ax.plot(actual_x, actual_y, 'b-', linewidth=2.5, alpha=0.9, label='实际路径', zorder=2)
    ax.scatter(actual_x[0], actual_y[0], c='blue', s=120, marker='o', edgecolors='white', label='实际起点', zorder=4)
    ax.scatter(actual_x[-1], actual_y[-1], c='blue', s=120, marker='s', edgecolors='white', label='实际终点', zorder=4)

    if frames:
        first_ego = frames[0]["ego_state"]
        last_ego = frames[-1]["ego_state"]
        ax.scatter(first_ego["x"], first_ego["y"], c='green', s=180, marker='o', 
                   edgecolors='black', linewidths=1.5, label='数据集起点', zorder=5)
        ax.scatter(last_ego["x"], last_ego["y"], c='green', s=180, marker='s', 
                   edgecolors='black', linewidths=1.5, label='数据集终点', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'实际路径 vs 参考轨迹\n日志: {os.path.basename(log_path)}')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 显示统计信息
    if errors:
        avg_err = np.mean(errors)
        max_err = np.max(errors)
        ax.text(0.02, 0.02, f'平均偏离: {avg_err:.2f} m\n最大偏离: {max_err:.2f} m',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图片
    save_dir = os.path.join(args.save_to_dir, 'path')
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    save_path_img = os.path.join(save_dir, f"{base_name}.png")
    plt.savefig(save_path_img, dpi=150)
    print(f"✅ 图片已保存: {save_path_img}")

    plt.show()

    # ===== 保存 CSV 对比列表 =====
    csv_path = os.path.join(save_dir, f"{base_name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['time', 'actual_x', 'actual_y', 'expected_x', 'expected_y', 'error_distance'])
        for t, ax, ay, ex, ey, err in zip(times, actual_x, actual_y, expected_x, expected_y, errors):
            writer.writerow([f"{t:.3f}", f"{ax:.6f}", f"{ay:.6f}", f"{ex:.6f}", f"{ey:.6f}", f"{err:.6f}"])
    print(f"✅ CSV 对比列表已保存: {csv_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()