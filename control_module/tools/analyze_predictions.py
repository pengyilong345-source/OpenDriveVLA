#!/usr/bin/env python
"""
离线分析 OpenDriveVLA 预测数据
计算每帧的 ADE/FDE，生成汇总报告
用法：python analyze_predictions.py
"""
import os
import json
import glob
import math
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "open_loop_pilot")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "prediction_analysis")

def compute_ade_fde(pred_traj, gt_traj):
    """计算 ADE 和 FDE（均方根误差）"""
    assert len(pred_traj) == len(gt_traj), "轨迹长度不一致"
    errors = []
    for p, gt in zip(pred_traj, gt_traj):
        dx = p[0] - gt[0]
        dy = p[1] - gt[1]
        errors.append(math.hypot(dx, dy))
    ade = sum(errors) / len(errors)
    fde = errors[-1]
    return ade, fde, errors

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集所有 predictions.json
    json_files = []
    for group in ["G1_official_local", "G2_complex_language"]:
        pattern = os.path.join(DATA_ROOT, group, "**", "predictions.json")
        for f in glob.glob(pattern, recursive=True):
            json_files.append((group, f))
    
    print(f"找到 {len(json_files)} 个 predictions.json 文件")
    
    # 存储结果
    results = []  # 每行一个样本
    scenario_stats = defaultdict(lambda: {"ade_list": [], "fde_list": [], "count": 0})
    
    for group, filepath in json_files:
        # 提取场景标识
        rel_path = os.path.relpath(filepath, os.path.join(DATA_ROOT, group))
        parts = rel_path.split(os.sep)
        scenario = parts[0] if len(parts) > 0 else "unknown"
        subscenario = parts[1] if len(parts) > 1 else "unknown"
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scenario_id = data.get("scenario_id", scenario)
        subscenario_name = data.get("subscenario", subscenario)
        
        for sample in data.get("samples", []):
            pred = sample.get("parsed_trajectory")
            gt = sample.get("gt_future_trajectory")
            if pred is None or gt is None or len(pred) != len(gt):
                continue
            if len(pred) == 0:
                continue
            
            # 计算误差
            ade, fde, errors = compute_ade_fde(pred, gt)
            tick = sample.get("tick", 0)
            sim_t = sample.get("sim_t", 0.0)
            instruction = sample.get("raw_instruction", "")
            
            results.append({
                "group": group,
                "scenario": scenario,
                "subscenario": subscenario_name,
                "tick": tick,
                "sim_t": sim_t,
                "ade": ade,
                "fde": fde,
                "instruction": instruction,
                "file": filepath
            })
            
            scenario_stats[(group, scenario, subscenario_name)]["ade_list"].append(ade)
            scenario_stats[(group, scenario, subscenario_name)]["fde_list"].append(fde)
            scenario_stats[(group, scenario, subscenario_name)]["count"] += 1
    
    # 保存汇总 CSV
    csv_path = os.path.join(OUTPUT_DIR, "prediction_errors.csv")
    # 修正 fieldnames，添加 'file'
    fieldnames = ["group","scenario","subscenario","tick","sim_t","ade","fde","instruction","file"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"✅ 保存详细结果至 {csv_path}")
    
    # 按场景统计
    summary_path = os.path.join(OUTPUT_DIR, "scenario_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["group","scenario","subscenario","count","ade_mean","ade_std","fde_mean","fde_std"])
        for (group, scenario, subscenario), stats in scenario_stats.items():
            ade_arr = np.array(stats["ade_list"])
            fde_arr = np.array(stats["fde_list"])
            writer.writerow([
                group, scenario, subscenario,
                stats["count"],
                np.mean(ade_arr), np.std(ade_arr),
                np.mean(fde_arr), np.std(fde_arr)
            ])
    print(f"✅ 保存场景汇总至 {summary_path}")
    
    # 绘制分组箱线图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    groups = ["G1_official_local", "G2_complex_language"]
    ade_data = []
    fde_data = []
    labels = []
    for group in groups:
        group_ades = [r["ade"] for r in results if r["group"] == group]
        group_fdes = [r["fde"] for r in results if r["group"] == group]
        ade_data.append(group_ades)
        fde_data.append(group_fdes)
        labels.append(group.replace("_", " "))
    
    axes[0].boxplot(ade_data, labels=labels)
    axes[0].set_title("ADE by Group")
    axes[0].set_ylabel("ADE (m)")
    axes[1].boxplot(fde_data, labels=labels)
    axes[1].set_title("FDE by Group")
    axes[1].set_ylabel("FDE (m)")
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "group_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"✅ 保存对比图至 {plot_path}")

if __name__ == "__main__":
    main()