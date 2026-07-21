#!/usr/bin/env python3
"""Evaluate existing mini predictions against native mini trajectory GT."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import numpy as np

def args_parser():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--gt-traj',type=Path,required=True); p.add_argument('--gt-mask',type=Path,required=True); p.add_argument('--tokens',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); return p.parse_args()
def load_tokens(path):
 d=json.loads(path.read_text()); return d['tokens'] if isinstance(d,dict) else d
def path_length(x):
 if len(x)==0:return 0.0
 return float(np.linalg.norm(x[0])+np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
def aggregate(rows):
 valid=[r for r in rows if r['parse_success'] and r['ade_m'] is not None]; waypoint=[]
 for i in range(6):
  vals=[r['per_waypoint_l2_m'][i] for r in valid if r['per_waypoint_l2_m'][i] is not None]; waypoint.append(float(np.mean(vals)) if vals else None)
 all_l2=[v for r in valid for v in r['per_waypoint_l2_m'] if v is not None]
 return {'sample_count':len(valid),'per_waypoint_l2_m':waypoint,'l2_at_1s_m':waypoint[1],'l2_at_2s_m':waypoint[3],'l2_at_3s_m':waypoint[5],
  'cumulative_l2_1s_m':float(np.mean(waypoint[:2])) if all(v is not None for v in waypoint[:2]) else None,
  'cumulative_l2_2s_m':float(np.mean(waypoint[:4])) if all(v is not None for v in waypoint[:4]) else None,
  'cumulative_l2_3s_m':float(np.mean(waypoint[:6])) if all(v is not None for v in waypoint[:6]) else None,
  'ade_m':float(np.mean(all_l2)) if all_l2 else None,'fde_m':float(np.mean([r['fde_m'] for r in valid])) if valid else None,
  'average_predicted_path_length_m':float(np.mean([r['predicted_path_length_m'] for r in valid])) if valid else None,
  'average_gt_path_length_m':float(np.mean([r['gt_path_length_m'] for r in valid])) if valid else None}
def main():
 a=args_parser(); tokens=load_tokens(a.tokens); pred_payload=json.loads(a.predictions.read_text()); pred={x.get('sample_token',x['token']):x for x in pred_payload['results']}
 gt=pickle.load(open(a.gt_traj,'rb')); masks=pickle.load(open(a.gt_mask,'rb')); selected=set(tokens); intersection=[t for t in tokens if t in pred and t in gt and t in masks]; rows=[]
 for token in intersection:
  result=pred[token]; parsed=result.get('parsed_trajectory'); parse_success=bool(result.get('parse_success',parsed is not None)) and parsed is not None
  g=np.asarray(gt[token],dtype=np.float64).reshape(-1,2); m=np.asarray(masks[token]).reshape(-1,2).any(axis=1); valid_idx=np.flatnonzero(m)
  p=None; l2=[None]*len(g); ade=fde=pred_len=None; all_zero=False
  if parse_success:
   p=np.asarray(parsed,dtype=np.float64)
   if p.shape!=(6,2) or not np.isfinite(p).all(): parse_success=False
  if parse_success:
   all_zero=bool(np.all(np.abs(p)<=1e-8)); distances=np.linalg.norm(p-g,axis=1); l2=[float(distances[i]) if m[i] else None for i in range(len(g))]
   values=[x for x in l2 if x is not None]; ade=float(np.mean(values)) if values else None; fde=float(distances[valid_idx[-1]]) if len(valid_idx) else None; pred_len=path_length(p)
  valid_gt=g[m]; gt_len=path_length(valid_gt)
  rows.append({'sample_token':token,'parse_success':parse_success,'all_zero_prediction':all_zero,'predicted_trajectory':parsed,'gt_trajectory':g.tolist(),
   'gt_mask':np.asarray(masks[token]).reshape(-1,2).tolist(),'valid_gt_point_count':int(m.sum()),'per_waypoint_l2_m':l2,
   'l2_at_1s_m':l2[1] if len(l2)>1 else None,'l2_at_2s_m':l2[3] if len(l2)>3 else None,'l2_at_3s_m':l2[5] if len(l2)>5 else None,
   'ade_m':ade,'fde_m':fde,'predicted_path_length_m':pred_len,'gt_path_length_m':gt_len})
 parse_count=sum(r['parse_success'] for r in rows); zeros=sum(r['all_zero_prediction'] for r in rows); agg=aggregate(rows)
 metrics={'experiment_label':'nuScenes-mini native-compatible trajectory evaluation','selected_sample_count':len(tokens),'prediction_count':len(pred),'gt_count':len(gt),
  'exact_token_intersection':intersection,'intersection_count':len(intersection),'missing_prediction_tokens':[t for t in tokens if t not in pred],
  'missing_gt_tokens':[t for t in tokens if t not in gt or t not in masks],'extra_prediction_tokens':[t for t in pred if t not in selected],
  'parse_success_count':parse_count,'parse_success_rate':parse_count/len(tokens) if tokens else 0.0,'all_zero_prediction_count':zeros,
  'all_zero_prediction_rate':zeros/parse_count if parse_count else 0.0,'valid_gt_count':sum(any(np.asarray(masks[t]).reshape(-1,2).any(axis=1)) for t in intersection),
  'metrics_including_parsed_all_zero_outputs':agg,'metrics_over_all_parse_success_samples':agg,'collision_metrics':'not_computed',
  'collision_reason':'No real mini-specific planning occupancy/segmentation GT is available.','parse_failure_policy':'not converted to zero; excluded from numeric metrics',
  'all_zero_policy':'retained and evaluated as a real model prediction'}
 a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'trajectory_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); (a.output_dir/'per_sample_metrics.json').write_text(json.dumps({'samples':rows},indent=2)+'\n')
 lines=['nuScenes-mini native-compatible trajectory evaluation',f'Selected/prediction/GT/intersection: {len(tokens)}/{len(pred)}/{len(gt)}/{len(intersection)}',
  f'Parse success: {parse_count}/{len(tokens)} ({metrics["parse_success_rate"]:.2%})',f'All-zero parsed predictions: {zeros}/{parse_count} ({metrics["all_zero_prediction_rate"]:.2%})',
  f'L2 @ 1s / 2s / 3s: {agg["l2_at_1s_m"]:.6f} / {agg["l2_at_2s_m"]:.6f} / {agg["l2_at_3s_m"]:.6f} m',
  f'ADE / FDE: {agg["ade_m"]:.6f} / {agg["fde_m"]:.6f} m',f'Average predicted / GT path length: {agg["average_predicted_path_length_m"]:.6f} / {agg["average_gt_path_length_m"]:.6f} m',
  'Collision: not_computed (no real mini occupancy GT)']
 text='\n'.join(lines)+'\n'; (a.output_dir/'evaluation_summary.txt').write_text(text); print(text)
if __name__=='__main__':main()
