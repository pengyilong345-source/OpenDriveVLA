import os
import json
import glob
from typing import List, Dict, Any


def load_frames(data_dir: str) -> List[Dict[str, Any]]:
    """
    加载指定目录下所有 frame_*.json 文件，按 frame_id 排序。
    """
    pattern = os.path.join(data_dir, "frame_*.json")
    file_paths = glob.glob(pattern)

    def extract_frame_id(path):
        base = os.path.basename(path)  # frame_000020.json
        num_str = base.split('_')[1].split('.')[0]  # '000020'
        return int(num_str)

    file_paths.sort(key=extract_frame_id)

    frames = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:   # 指定 UTF-8 编码
            data = json.load(f)
        frames.append(data)
    return frames