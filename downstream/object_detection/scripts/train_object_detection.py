from .detectron_utils import *


def train_with_default_trainer(cfg, resume=True):
    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([MLflowHook(cfg)])
    trainer.resume_or_load(resume=resume)
    trainer.train()


def train_with_custom_trainer(cfg, resume=True):
    trainer = CustomTrainer(cfg)
    trainer.register_hooks([MLflowHook(cfg)])
    trainer.resume_or_load(resume=resume)
    trainer.train()