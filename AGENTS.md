# Project Instructions for Codex

## Project goal

We are adapting OpenDriveVLA to run CARLA simulation experiments.

Current stage:
We are NOT connecting to a real CARLA server yet.
We are NOT implementing speech recognition yet.
We are NOT modifying the original OpenDriveVLA model architecture yet.

The current goal is to create and validate a minimal CARLA-style mock data pipeline:

1. Create one mock CARLA sample.
2. Save six dummy camera images.
3. Save a CARLA-style metadata file:
   `/root/autodl-tmp/workspace/data/carla/infos/carla_infos_val.pkl`
4. Implement `CarlaLLaVADataset`.
5. Verify that the dataset can read the mock sample, load images, and build a driving prompt.
6. Do not run model inference in this stage.

## Repository status

- This repository is a fork of OpenDriveVLA.
- The active development branch is `carla_voice_vla`.
- OpenDriveVLA-0.5B checkpoint can already be loaded successfully.
- The official nuScenes eval currently stops because nuScenes data is missing.
- We are now building a CARLA data adapter.

## Important paths

Project root:
`/root/autodl-tmp/workspace/OpenDriveVLA`

CARLA mock data root:
`/root/autodl-tmp/workspace/data/carla`

CARLA info file:
`/root/autodl-tmp/workspace/data/carla/infos/carla_infos_val.pkl`

Checkpoint:
`/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B`

## Implementation rules

- Prefer adding new files under `carla_vla/`.
- Do not modify the original model architecture.
- Do not modify `drivevla/inference_drivevla.py` in this stage.
- Do not depend on a running CARLA server in this stage.
- Do not use nuScenes APIs in the CARLA mock dataset.
- Keep the code simple and readable.
- Use Python 3.10.
- Use PIL for creating and loading mock images.
- The dataset should be easy to extend later for real CARLA data.

## Files to create in this stage

Create these files:

- `carla_vla/__init__.py`
- `carla_vla/data_utils/__init__.py`
- `carla_vla/data_utils/carla_llava_dataset.py`
- `carla_vla/tools/create_mock_carla_data.py`
- `carla_vla/tools/test_carla_dataset.py`

## Mock CARLA data format

The mock info file should be a pickle file containing a list with one sample.

Each sample should contain:

- `sample_id`
- `timestamp`
- `images`
- `ego`
- `agents`
- `map`
- `weather`
- `command`

The image dictionary should contain six cameras:

- `CAM_FRONT`
- `CAM_FRONT_LEFT`
- `CAM_FRONT_RIGHT`
- `CAM_BACK`
- `CAM_BACK_LEFT`
- `CAM_BACK_RIGHT`

Each image path should be relative to:
`/root/autodl-tmp/workspace/data/carla`

Example:

```python
"images": {
    "CAM_FRONT": "images/mock_000000/CAM_FRONT.png",
    "CAM_FRONT_LEFT": "images/mock_000000/CAM_FRONT_LEFT.png",
    "CAM_FRONT_RIGHT": "images/mock_000000/CAM_FRONT_RIGHT.png",
    "CAM_BACK": "images/mock_000000/CAM_BACK.png",
    "CAM_BACK_LEFT": "images/mock_000000/CAM_BACK_LEFT.png",
    "CAM_BACK_RIGHT": "images/mock_000000/CAM_BACK_RIGHT.png"
}

## Default command

The default command should be:

Drive safely and follow the lane.

## Dataset requirements

CarlaLLaVADataset should:

- Read `carla_infos_val.pkl`.
- Load six camera images using PIL.
- Build a structured driving prompt from:
  - command
  - ego state
  - nearby agents
  - map information
  - weather
- Return a dictionary containing:
  - sample_id
  - prompt
  - images
  - ego
  - agents
  - map
  - weather
  - command

In this stage, the dataset does not need to tokenize the prompt.
In this stage, the dataset does not need to call OpenDriveVLA.
In this stage, the dataset only needs to prove that CARLA-style data can be loaded correctly.

## Test script requirements

`carla_vla/tools/test_carla_dataset.py` should:

- Load `CarlaLLaVADataset`.
- Read the mock info file.
- Print:
  - dataset length
  - sample_id
  - prompt
  - number of images
  - image sizes
  - ego state
  - map info
  - first few agents
- Assert that:
  - dataset length is 1
  - there are 6 images
  - all images can be loaded
  - prompt is a non-empty string

## Do not commit

Do not commit:

- `data/`
- `checkpoints/`
- `output/`
- `outputs/`
- `logs/`
- `*.safetensors`
- `*.pth`
- `*.pt`
- `*.bin`
- `*.mp4`
- `*.zip`
- `*.tar`