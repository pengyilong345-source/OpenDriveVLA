#!/usr/bin/env python
"""
绘制预测回放的实际路径，并与 predictions.json 中的预测轨迹对比
用法：
    python plot_prediction_replay.py --log logs/prediction_replay_20260723_112108.csv --json open_loop_pilot/G2_complex_language/s3_3_temp_pedestrian_crossing/seed303/predictions.json
"""
import os
import sys
import argparse
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_predictions(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    samples = data.get("samples", [])
    traj_list = []
    for s in samples:
        traj = s.get("parsed_trajectory")
        if traj:
            traj_list.append(traj)
    return traj_list

def main():
    parser = argparse.ArgumentParser(description="绘制预测回放路径与预测轨迹对比")
    parser.add_argument("--log", required=True, help="CSV日志文件路径")
    parser.add_argument("--json", help="predictions.json 文件路径（可选，若提供则绘制预测轨迹）")
    parser.add_argument("--output", help="输出图片路径（可选）")
    args = parser.parse_args()
    
    df = pd.read_csv(args.log)
    print(f"加载日志: {args.log}, 共 {len(df)} 帧")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 实际路径（世界坐标）
    x = df['x'].values
    y = df['y'].values
    # 以起点为原点进行平移（便于与预测点对齐）
    x0, y0 = x[0], y[0]
    x_rel = x - x0
    y_rel = y - y0
    
    ax.plot(x_rel, y_rel, 'b-', linewidth=2, label='Actual Path')
    ax.scatter(x_rel[0], y_rel[0], c='green', s=100, label='Start')
    ax.scatter(x_rel[-1], y_rel[-1], c='red', s=100, label='End')
    
    # 如果提供了 json，绘制预测轨迹点
    if args.json:
        traj_list = load_predictions(args.json)
        print(f"加载预测轨迹: {len(traj_list)} 帧")
        # 为每个样本绘制不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(traj_list)))
        for i, traj in enumerate(traj_list):
            # 预测点是相对坐标（自车坐标系），直接使用
            if traj:
                pts = np.array(traj)
                # 如果轨迹有多个点，画线连接
                ax.plot(pts[:, 0], pts[:, 1], '--', color=colors[i], linewidth=1.5, alpha=0.7,
                        label=f'Prediction frame {i+1}')
                # 画点
                ax.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=30, alpha=0.8)
        ax.legend()
    
    ax.set_xlabel('Relative X (m)')
    ax.set_ylabel('Relative Y (m)')
    ax.set_title('Actual Path vs Prediction Trajectories')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"✅ 路径图保存至 {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()