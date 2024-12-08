from detectron2.engine import DefaultPredictor
from .detectron_utils import *


def do_coco_eval(config_file, detectron_output_dir):
    cfg = get_test_cfg(config_file)
    cfg.defrost()
    cfg.OUTPUT_DIR = detectron_output_dir
    cfg.MODEL.WEIGHTS = os.path.join(detectron_output_dir, "model_final.pth")
    cfg.freeze()
    test_dataset = cfg.DATASETS.TEST[0]
    logger.info(f"Using {test_dataset} dataset for evaluating the final model.")
    predictor = DefaultPredictor(cfg)
    coco_evaluator(cfg, predictor, test_dataset)
    test_image(test_dataset, predictor, n=5, threshold=0.5)
