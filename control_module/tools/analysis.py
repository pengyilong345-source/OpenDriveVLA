"""
分析脚本：读取 CARLA 控制日志，绘制性能曲线并计算统计指标。

用法：
    1. 默认分析最新的日志：
        python analysis.py

    2. 分析指定的一个或多个日志（用于对比）：
        python analysis.py logs/carla_control_20260712_235416.csv logs/carla_control_20260713_001122.csv

    3. 指定轨迹文件（默认使用 tests/test_data/mock_trajectories.json）：
        python analysis.py --trajectory path/to/trajectory.json logs/xxx.csv
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_log(log_dir='../logs'):
    """查找目录下最新的 CSV 日志文件"""
    log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    # files = glob.glob(os.path.join(log_dir, 'carla_control_*.csv'))
    files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
    if not files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
    latest = max(files, key=os.path.getmtime)
    return latest


def load_trajectory(json_path):
    """加载轨迹 JSON，返回时间数组和对应的 x, y 坐标（插值到 0.01s 间隔）"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    pts = data['future_trajectory_ego_frame']
    # 提取 dt, x, y，注意 dt 可能是浮点数
    times = [p['dt'] for p in pts]
    xs = [p['x'] for p in pts]
    ys = [p['y'] for p in pts]

    # 插值到更密集的时间点（用于显示和误差计算）
    dt_min = 0.01
    t_new = np.arange(0, times[-1] + dt_min, dt_min)
    # 线性插值
    f_x = interp1d(times, xs, kind='linear', fill_value='extrapolate')
    f_y = interp1d(times, ys, kind='linear', fill_value='extrapolate')
    return t_new, f_x(t_new), f_y(t_new)


def compute_errors(actual_x, actual_y, target_t, target_x, target_y):
    """
    计算实际路径与目标轨迹的误差。
    返回：横向误差（法向）、纵向误差（切向）、最近距离误差（欧氏距离）
    """
    errors = []
    lateral_errors = []
    longitudinal_errors = []
    for ax, ay in zip(actual_x, actual_y):
        # 计算实际点与目标轨迹上所有点的距离
        dists = np.hypot(ax - target_x, ay - target_y)
        idx = np.argmin(dists)
        # 最近点坐标
        tx, ty = target_x[idx], target_y[idx]
        # 计算切向单位向量（用前后点差分）
        if idx == 0:
            dx, dy = target_x[1] - target_x[0], target_y[1] - target_y[0]
        elif idx == len(target_x) - 1:
            dx, dy = target_x[-1] - target_x[-2], target_y[-1] - target_y[-2]
        else:
            dx, dy = target_x[idx+1] - target_x[idx-1], target_y[idx+1] - target_y[idx-1]
        norm = np.hypot(dx, dy)
        if norm > 1e-6:
            dx, dy = dx/norm, dy/norm
        else:
            dx, dy = 1.0, 0.0
        # 偏差向量
        ex, ey = ax - tx, ay - ty
        # 横向误差 = 偏差向量在法线方向投影（法线为(-dy, dx)）
        lat = ex * (-dy) + ey * dx
        # 纵向误差 = 偏差向量在切线方向投影
        lon = ex * dx + ey * dy
        lateral_errors.append(lat)
        longitudinal_errors.append(lon)
        errors.append(np.hypot(ex, ey))
    return np.array(lateral_errors), np.array(longitudinal_errors), np.array(errors)


def analyze_log(csv_path, traj_json_path=None, label=None):
    """分析单个日志文件，返回绘图数据和统计"""
    # 读取 CSV
    df = pd.read_csv(csv_path)
    time = df['time'].values
    x = df['x'].values
    y = df['y'].values
    speed = df['speed'].values
    steer = df['steer'].values

    # 如果没有指定轨迹文件，尝试从默认位置加载
    if traj_json_path is None:
        default_json = os.path.join(os.path.dirname(__file__), '../tests/test_data/mock_trajectories.json')
        if os.path.exists(default_json):
            traj_json_path = default_json
        else:
            raise FileNotFoundError("未指定轨迹文件，且默认路径不存在")

    # 加载目标轨迹
    t_target, x_target, y_target = load_trajectory(traj_json_path)

    # 计算误差（实际点与目标轨迹的最近距离）
    lat_err, lon_err, dist_err = compute_errors(x, y, t_target, x_target, y_target)

    # 计算统计指标
    stats = {
        'time_range': (time[0], time[-1]),
        'speed_mean': np.mean(speed),
        'speed_std': np.std(speed),
        'steer_mean': np.mean(steer),
        'steer_std': np.std(steer),
        'lat_err_mean': np.mean(lat_err),
        'lat_err_std': np.std(lat_err),
        'lat_err_max': np.max(np.abs(lat_err)),
        'dist_err_mean': np.mean(dist_err),
        'dist_err_std': np.std(dist_err),
        'dist_err_max': np.max(dist_err),
    }

    # 准备绘图数据
    plot_data = {
        'time': time,
        'x': x,
        'y': y,
        'speed': speed,
        'steer': steer,
        'lat_err': lat_err,
        'dist_err': dist_err,
        't_target': t_target,
        'x_target': x_target,
        'y_target': y_target,
        'label': label if label else os.path.basename(csv_path).replace('.csv', '')
    }
    return plot_data, stats


def plot_single(plot_data, stats, save_path=None):
    """绘制单个日志的分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'控制性能分析 - {plot_data["label"]}', fontsize=16)

    # 1. 横向误差随时间变化
    ax = axes[0, 0]
    ax.plot(plot_data['time'], plot_data['lat_err'], 'b-', lw=1.5)
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('横向误差 (m)')
    ax.set_title('横向误差（法向偏差）')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f'平均: {stats["lat_err_mean"]:.3f} m\n最大: {stats["lat_err_max"]:.3f} m',
            transform=ax.transAxes, verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 转向角随时间变化
    ax = axes[0, 1]
    ax.plot(plot_data['time'], plot_data['steer'], 'g-', lw=1.5)
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('转向角 (归一化)')
    ax.set_title('转向角控制')
    ax.grid(True, alpha=0.3)

    # 3. 速度跟踪
    ax = axes[1, 0]
    ax.plot(plot_data['time'], plot_data['speed'], 'r-', lw=1.5, label='实际速度')
    # 添加目标速度线（如果有的话）
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('速度跟踪')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. 实际路径 vs 目标轨迹
    ax = axes[1, 1]
    ax.plot(plot_data['x_target'], plot_data['y_target'], 'b--', lw=2, label='目标轨迹')
    ax.plot(plot_data['x'], plot_data['y'], 'r-', lw=1.5, label='实际路径')
    ax.scatter(plot_data['x'][0], plot_data['y'][0], c='g', s=80, marker='o', label='起点')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('路径跟踪对比')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存至: {save_path}")
    plt.show()


def plot_compare(plot_data_list, save_path=None):
    """对比多个日志"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('多测试对比分析', fontsize=16)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # 横向误差对比
    ax = axes[0, 0]
    for i, data in enumerate(plot_data_list):
        ax.plot(data['time'], data['lat_err'], color=colors[i % len(colors)], lw=1.5, label=data['label'])
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('横向误差 (m)')
    ax.set_title('横向误差对比')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 转向角对比
    ax = axes[0, 1]
    for i, data in enumerate(plot_data_list):
        ax.plot(data['time'], data['steer'], color=colors[i % len(colors)], lw=1.5, label=data['label'])
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('转向角')
    ax.set_title('转向角对比')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 速度对比
    ax = axes[1, 0]
    for i, data in enumerate(plot_data_list):
        ax.plot(data['time'], data['speed'], color=colors[i % len(colors)], lw=1.5, label=data['label'])
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('速度对比')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 路径对比
    ax = axes[1, 1]
    for i, data in enumerate(plot_data_list):
        ax.plot(data['x'], data['y'], color=colors[i % len(colors)], lw=1.5, label=data['label'])
    # 画目标轨迹
    if plot_data_list:
        t0 = plot_data_list[0]
        ax.plot(t0['x_target'], t0['y_target'], 'k--', lw=2, label='目标轨迹')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('路径对比')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"对比图已保存至: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='分析 CARLA 控制日志')
    parser.add_argument('logs', nargs='*', help='日志文件路径（支持多个）')
    parser.add_argument('--trajectory', '-t', help='轨迹 JSON 文件路径')
    parser.add_argument('--output', '-o', help='保存图表的路径（不指定则显示）')
    args = parser.parse_args()

    # 确定日志文件列表
    if args.logs:
        log_paths = args.logs
    else:
        try:
            latest = find_latest_log()
            log_paths = [latest]
            print(f"自动选择最新日志: {latest}")
        except FileNotFoundError as e:
            print(e)
            return

    # 分析每个日志
    plot_data_list = []
    for i, path in enumerate(log_paths):
        try:
            label = f'Test {i+1}' if len(log_paths) > 1 else None
            data, stats = analyze_log(path, args.trajectory, label)
            plot_data_list.append(data)
            # 打印统计信息
            print(f"\n=== 分析: {data['label']} ===")
            for key, val in stats.items():
                if isinstance(val, float):
                    print(f"{key}: {val:.4f}")
                else:
                    print(f"{key}: {val}")
        except Exception as e:
            print(f"处理 {path} 时出错: {e}")

    if not plot_data_list:
        print("没有成功分析任何日志。")
        return

    # 绘图
    if len(plot_data_list) == 1:
        plot_single(plot_data_list[0], stats, args.output)
    else:
        plot_compare(plot_data_list, args.output)


if __name__ == '__main__':
    main()