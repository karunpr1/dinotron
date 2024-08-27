from .detectron_utils import *


def train_with_default_trainer(cfg, resume=True):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


def train_with_custom_trainer(cfg, resume=True):
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


def train_with_student_network(cfg, resume=True):
    trainer = CustomTrainerStudent(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


def train_with_teacher_network(cfg, resume=True):
    trainer = CustomTrainerTeacher(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()
