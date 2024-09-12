from dataclasses import dataclass
from typing import Tuple

@dataclass

@dataclass
class Params:
    # Model Architecture
    arch: str = "resnet50"  # Options: 'vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'
    patch_size: int = 16
    out_dim: int = 65536
    norm_last_layer: bool = True
    momentum_teacher: float = 0.996
    use_bn_in_head: bool = False
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 0
    use_fp16: bool = True

    # Optimization Hyperparameters
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    batch_size_per_gpu: int = 64
    epochs: int = 100
    freeze_last_layer: int = 1
    lr: float = 0.0005
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    optimizer: str = "adamw"  # Options: 'adamw', 'sgd', 'lars'
    drop_path_rate: float = 0.1

    # Crop Scales
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_number: int = 8
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)

    # Data and Training Path
    data_path: str = "/path/to/train/"
    output_dir: str = "./output"
    saveckp_freq: int = 20

    # Miscellaneous
    seed: int = 42
    num_workers: int = 10
    dist_url: str = "env://"
    local_rank: int = 0


@dataclass
class DinoConfig:
    params: Params