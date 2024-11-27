from dataclasses import dataclass


@dataclass
class Params:
    final_model_name: str
    pretrain_method: str
    backbone: str
    trainer: str
    max_epochs: int
    resume: bool
    device: str
    devices: str


@dataclass
class Paths:
    merge_config_file: str
    pretrained_weights: str
    train_image_path: str
    train_annotations_file: str
    test_image_path: str
    test_annotations_file: str
    output_dir: str
    config_save_name: str
    pickle_config_file_path: str


@dataclass
class Data:
    dataset_name: str
    label_fraction: str
    num_classes: int
    num_workers: int
    classes: list
    register_dataset: str


@dataclass
class Model:
    score_thresh_test: float
    batch_size_per_image: int
    backbone_freeze_at: int


@dataclass
class Solver:
    img_per_batch: int
    base_lr: float
    max_iters: int
    steps: list
    gamma: float
    warmup_iters: int


@dataclass
class DetectronConfig:
    params: Params
    paths: Paths
    data: Data
    model: Model
    solver: Solver
