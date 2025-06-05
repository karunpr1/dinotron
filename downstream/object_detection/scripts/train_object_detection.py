from .detectron_utils import *


def train_with_default_trainer(cfg, resume=True):
    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([MLflowHook(cfg)])
    trainer.resume_or_load(resume=resume)
    trainer.train()


def train_with_custom_trainer(cfg, resume=True):
    # cfg.defrost()
    # cfg.SOLVER.OPTIMIZER = "AdamW" # SGD
    # cfg.SOLVER.WEIGHT_DECAY = 0.01
    # cfg.SOLVER.AMSGRAD = False
    # cfg.SOLVER.WARMUP_FACTOR = 0.001
    # cfg.SOLVER.WARMUP_METHOD = "linear"
    # cfg.freeze()
    trainer = CustomTrainer(cfg)
    trainer.register_hooks([MLflowHook(cfg)])
    trainer.resume_or_load(resume=resume)
    trainer.train()