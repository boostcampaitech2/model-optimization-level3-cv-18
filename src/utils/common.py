from typing import Any, Dict, Union

import numpy as np
import yaml
import random
import os
import torch

from torchvision.datasets import ImageFolder, VisionDataset


def read_yaml(cfg: Union[str, Dict[str, Any]]):
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = cfg
    return config


def get_label_counts(dataset_path: str):
    """Counts for each label."""
    if not dataset_path:
        return None
    td = ImageFolder(root=dataset_path)
    # get label distribution
    label_counts = [0] * len(td.classes)
    for p, l in td.samples:
        label_counts[l] += 1
    return label_counts


def seed_everything(seed: int = 21):
    """Seed 고정으로 일관된 실험 결과를 얻는다."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["PYTHONHASHSEED"] = str(seed)

