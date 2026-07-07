# OpenDriveVLA Environment Notes

Verified on AutoDL RTX 4090.

Key versions:
- torch: 2.1.2+cu118
- transformers: 4.49.0
- huggingface_hub: 0.29.3
- deepspeed: 0.12.6
- mmcv-full: 1.7.2
- mmdet: 2.26.0
- mmsegmentation: 0.29.1
- numpy: 1.26.4

Status:
- mmcv/mmdet3d compiled successfully.
- OpenDriveVLA-0.5B checkpoint downloaded.
- Model loading succeeded.
- Official eval currently stops because nuScenes file is missing:
  data/infos/nuscenes_infos_temporal_val.pkl
