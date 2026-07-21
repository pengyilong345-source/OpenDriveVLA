#!/usr/bin/env python3
"""Dump a compact schema of the target-free mini batch before generate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from carla_vla.data_utils.nuscenes_mini_inference_adapter import CAMERA_ORDER, IMG_MEAN_BGR, NuScenesMiniInferenceAdapter
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token

GT_KEYS = {"sdc_planning","sdc_planning_mask","gt_segmentation","gt_instance","gt_lane_labels","gt_lane_masks","gt_boxes","gt_names","gt_velocity","fut_traj","fut_traj_valid_mask","visibility_tokens","evaluation_targets"}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--info',type=Path,required=True); p.add_argument('--dataroot',type=Path,required=True)
    p.add_argument('--tokens',type=Path,required=True); p.add_argument('--model-path',type=Path,default=Path('/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B'))
    p.add_argument('--output',type=Path,default=Path('output/nuscenes_mini_drivevla/mini_runtime_batch_schema.json'))
    return p.parse_args()

def desc(value):
    if isinstance(value,torch.Tensor): return {'python_type':'torch.Tensor','shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
    if isinstance(value,np.ndarray): return {'python_type':'numpy.ndarray','shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,list): return {'python_type':'list','length':len(value),'items':[desc(x) for x in value[:2]],'items_truncated':len(value)>2}
    if isinstance(value,dict): return {'python_type':'dict','keys':list(value),'values':{k:desc(v) for k,v in value.items() if k not in ('img',)}}
    return {'python_type':type(value).__name__,'value':value if isinstance(value,(str,int,float,bool,type(None))) else repr(value)}

def nested_keys(value):
    keys=set()
    if isinstance(value,dict):
        keys.update(value)
        for item in value.values(): keys.update(nested_keys(item))
    elif isinstance(value,(list,tuple)):
        for item in value: keys.update(nested_keys(item))
    return keys

def main():
    a=parse_args(); wanted=json.loads(a.tokens.read_text())['tokens']; dataset=NuScenesMiniInferenceAdapter(a.info,a.dataroot)
    index=next(i for i,x in enumerate(dataset.infos) if x['token']==wanted[0]); sample=dataset[index]; data=sample['uniad_data']; meta=data['img_metas'][0][0]
    conv=conv_templates['qwen_planning_oriented_vlm'].copy(); conv.clear_conversation(); conv.append_message(conv.roles[0],sample['prompt']); conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt(); tokenizer=AutoTokenizer.from_pretrained(a.model_path,use_fast=False)
    ids=tokenizer_uniad_token(prompt,tokenizer,return_tensors='pt').unsqueeze(0)
    call={'input_ids':ids,'uniad_data':data,'do_sample':False,'temperature':0,'max_new_tokens':64,'num_beams':1}
    seen=nested_keys(call); excluded=sorted(GT_KEYS-seen); leaked=sorted(GT_KEYS & seen)
    can=np.asarray(meta['can_bus'])
    report={
      'sample_token':sample['token'],'scene_token':meta['scene_token'],'scene_name':sample['route_command']['source'].split('/')[1].replace('_route.json',''),
      'frame_idx':dataset.infos[index]['frame_idx'],'timestamp':sample['timestamp'],
      'top_level_generate_keys':list(call),'top_level_schema':{k:desc(v) for k,v in call.items() if k!='uniad_data'},
      'uniad_data_keys':list(data),'uniad_data_schema':{k:desc(v) for k,v in data.items() if k not in ('img','img_metas')},
      'image_tensor':desc(data['img'][0]),'generate_device_after_move':'cuda','camera_order':list(CAMERA_ORDER),
      'image_paths':[sample['image_paths'][c] for c in CAMERA_ORDER], 'original_image_shapes':[list(x) for x in meta['ori_shape']],
      'transformed_image_tensor_shape':list(data['img'][0].shape),'image_transform':{'color_input':'RGB PIL','model_color':'BGR','resize':None,'crop':None,'normalization_mean_bgr':IMG_MEAN_BGR.tolist(),'normalization_std_bgr':[1.0,1.0,1.0],'padding_divisor':32,'pad_shape':[list(x) for x in meta['pad_shape']]},
      'img_metas_keys':list(meta),'img_metas_schema':{k:desc(v) for k,v in meta.items()},
      'lidar2img_camera_count':len(meta['lidar2img']),'lidar2img_shapes':[list(np.asarray(x).shape) for x in meta['lidar2img']],
      'can_bus':{'python_type':type(meta['can_bus']).__name__,'shape':list(can.shape),'dtype':str(can.dtype),'values':can.tolist()},
      'ego_state_representation':{'global_translation':can[:3].tolist(),'orientation_wxyz':can[3:7].tolist(),'acceleration_xyz':can[7:10].tolist(),'rotation_rate_xyz':can[10:13].tolist(),'velocity_xyz':can[13:16].tolist(),'yaw_radians':float(can[16]),'yaw_degrees':float(can[17])},
      'historical_trajectory_representation':'unavailable in prompt; no history tensor is passed','navigation_command':sample['route_command'],
      'exact_prompt_text':prompt,'temporal_queue_length':1,'previous_bev_state_available':False,
      'cached_info_status':{'cached_info_used':False,'explicitly_bypassed':True},
      'gt_evaluation_keys_excluded':excluded,'gt_evaluation_keys_present':leaked,'gt_leakage_check_passed':not leaked,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote runtime batch schema to {a.output}; gt_leakage_check_passed={not leaked}')
if __name__=='__main__': main()
