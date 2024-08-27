from detectron2.engine import DefaultPredictor
from .detectron_utils import *


def do_coco_eval(config_file):
    cfg = get_test_cfg(config_file)
    test_dataset = cfg.DATASETS.TEST[0]
    logger.info(f"Using {test_dataset} dataset for evaluating the final model.")
    predictor = DefaultPredictor(cfg)
    test_image(test_dataset, predictor, n=5, threshold=0.5)
    coco_evaluator(cfg, predictor, test_dataset)
