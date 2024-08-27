from detectron2.engine import DefaultPredictor
from .detectron_utils import *


def do_coco_eval(config_file):
    cfg = get_test_cfg(config_file)
    logger.info(f"Using {cfg.DATASETS.TEST} dataset for evaluating the final model.")
    predictor = DefaultPredictor(cfg)
    test_image(cfg.DATASETS.TEST, predictor, n=5, threshold=0.5)
    coco_evaluator(cfg, predictor, cfg.DATASETS.TEST)
