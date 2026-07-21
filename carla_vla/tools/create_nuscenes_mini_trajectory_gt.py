#!/usr/bin/env python3
"""Create six-point trajectory GT for an exact native mini token list."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from mmdet3d.core.bbox import Box3DMode
from projects.mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj
from carla_vla.data_utils.nuscenes_mini_inference_adapter import NuScenesMiniInferenceAdapter

EPSILON_METRES=0.2

def args_parser():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataroot',type=Path,required=True); p.add_argument('--version',default='v1.0-mini')
 p.add_argument('--info',type=Path,required=True); p.add_argument('--tokens',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--future-steps',type=int,default=6); return p.parse_args()

def token_list(path):
 value=json.loads(path.read_text()); return value['tokens'] if isinstance(value,dict) else value

def atomic_pickle(path,value):
 tmp=path.with_name(path.name+'.tmp'); path.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('wb') as f: pickle.dump(value,f,pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)

def atomic_json(path,value):
 tmp=path.with_name(path.name+'.tmp'); path.parent.mkdir(parents=True,exist_ok=True); tmp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(tmp,path)

def main():
 a=args_parser()
 if a.version!='v1.0-mini' or a.future_steps<1: raise SystemExit('Only v1.0-mini and positive future steps are supported')
 tokens=token_list(a.tokens); payload=pickle.load(open(a.info,'rb')); infos={x['token']:x for x in payload['infos']}
 if list(infos) != tokens: raise RuntimeError('Token file must exactly match mini info order')
 nusc=NuScenes(version=a.version,dataroot=str(a.dataroot),verbose=False); native_tokens={x['token'] for x in nusc.sample}
 adapter=NuScenesMiniInferenceAdapter(a.info,a.dataroot); scene_names={x['token']:x['name'] for x in nusc.scene}
 traj_api=NuScenesTraj(nusc,predict_steps=12,planning_steps=a.future_steps,past_steps=4,fut_steps=4,with_velocity=True,
  CLASSES=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone'],
  box_mode_3d=Box3DMode.LIDAR,use_nonlinear_optimizer=True)
 trajs={}; masks={}; summaries=[]
 for token in tokens:
  if token not in native_tokens: raise KeyError(f'{token} is not v1.0-mini')
  info=infos[token]; current=nusc.get('sample',token)
  official_planning,official_mask,_=traj_api.get_sdc_planning_label(token)
  trajectory=official_planning[:,:,:2].astype(np.float32); mask=official_mask.astype(np.float32)
  if not np.isfinite(trajectory).all() or not np.isfinite(mask).all(): raise ValueError(f'NaN/Inf for {token}')
  valid=int(mask[0].any(axis=1).sum())
  command,command_detail=adapter.route_command(info); valid_xy=trajectory[0,:valid]; final=float(np.linalg.norm(valid_xy[-1])) if valid else 0.0
  path_length=float(np.linalg.norm(valid_xy[0])) if valid else 0.0
  if valid>1: path_length += float(np.linalg.norm(np.diff(valid_xy,axis=0),axis=1).sum())
  trajs[token]=trajectory; masks[token]=mask
  summaries.append({'sample_token':token,'scene_token':info['scene_token'],'scene_name':scene_names[info['scene_token']],
   'timestamp':info['timestamp'],'current_ego_speed_mps':float(np.linalg.norm(info['can_bus'][13:16])),
   'future_valid_point_count':valid,'gt_trajectory':trajectory[0].tolist(),'gt_mask':mask[0].tolist(),
   'gt_final_displacement_m':final,'gt_total_path_length_m':path_length,'near_stationary_epsilon_m':EPSILON_METRES,
   'gt_all_zero_or_near_stationary':valid>0 and final<EPSILON_METRES,'current_command':command_detail,
   'current_frame_index':info['frame_idx']})
 schema={'version':a.version,'source':'NuScenesTraj.get_sdc_planning_label over native v1.0-mini tables','token_count':len(tokens),'future_steps':a.future_steps,
  'time_interval_seconds':0.5,'reference_frame':'current LIDAR_TOP','trajectory_semantics':'absolute future LIDAR-origin offsets in current LIDAR coordinates',
  'trajectory_shape_per_token':[1,a.future_steps,2],'trajectory_dtype':'float32','mask_shape_per_token':[1,a.future_steps,2],
  'mask_dtype':'float32','mask_semantics':'1 for native future sample, 0 for unavailable; invalid stored zeros are not stationary GT',
  'near_stationary_epsilon_m':EPSILON_METRES,'all_finite':True,'gt_used_for_inference':False}
 atomic_pickle(a.output_dir/'gt_traj_mini.pkl',trajs); atomic_pickle(a.output_dir/'gt_traj_mask_mini.pkl',masks)
 atomic_json(a.output_dir/'gt_schema_report.json',schema); atomic_json(a.output_dir/'gt_sample_summary.json',{'samples':summaries})
 print(f'Generated native mini GT for {len(trajs)} tokens; valid counts={[x["future_valid_point_count"] for x in summaries]}')
if __name__=='__main__': main()
