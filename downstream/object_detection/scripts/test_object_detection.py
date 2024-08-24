from detectron2.engine import DefaultPredictor

import argparse
import os
import pickle
from detectron_utils import *
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('test_object_detection', add_help=False)
    parser.add_argument('--config', default='', type=str,
                        help="""config file which contains parameters for the object detection trainer""")
    # TODO: add other args to be passed in for the object detection trainer
    return parser


def test_object_detection(args):
    cfg = get_test_cfg(args.config_file_path)
    print(cfg)

    predictor = DefaultPredictor(cfg)
    test_image(cfg.test_dataset_name, predictor, n=5, threshold=0.5)
    coco_evaluator(cfg, predictor, f'{args.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_object_detection', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test_object_detection(args)
