from detectron2.utils.logger import setup_logger
setup_logger('detectron2')
from detectron2.engine import DefaultTrainer
from detectron_utils import *


import argparse
import os
import pickle
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('train_object_detection', add_help=False)
    parser.add_argument('--config', default='', type=str,
                       help="""config file which contains parameters for the object detection trainer""")
    # TODO: add other args to be passed in for the object detection trainer
    return parser


def train_object_detection(args):
    cfg = get_train_cfg(args.config_file_path, args.pretrained_weights, args.train_dataset_name, args.test_dataset_name,
                        args.num_classes, args.device, args.output_dir)
    print()

    with open(os.path.join(cfg.OUTPUT_DIR, f'{args.name}'), 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train_object_detection', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_object_detection(args)
