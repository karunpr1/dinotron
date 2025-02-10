from detectron2.engine import DefaultPredictor
from .detectron_utils import *


def eval_model(config_file, detectron_output_dir, test_dataset_name):
    cfg = get_test_cfg(config_file)
    cfg.defrost()
    cfg.OUTPUT_DIR = detectron_output_dir
    cfg.MODEL.WEIGHTS = os.path.join(detectron_output_dir, "model_final.pth")
    cfg.freeze()
    logger.info(f"Using {test_dataset_name} dataset for evaluating the final model.")
    predictor = DefaultPredictor(cfg)
    evaluation_results = coco_evaluator(cfg, predictor, test_dataset_name)
    for k, v in evaluation_results["bbox"].items():
        mlflow.log_metric(f"Test Set {k}", v, step=0)
    # mlflow.log_artifacts(detectron_output_dir, "test-set-evaluation")
    mlflow.log_text(str(evaluation_results), "test-set-evaluation/coco-metrics.txt")
    # test_image(test_dataset_name, predictor, n=5, threshold=0.5)
