import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from .detectron_utils import *
from .eval_utils import evaluate_model_multithresh, evaluate_model_multithresh_per_class
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import yaml
from tqdm import tqdm


def eval_model(config_file, detectron_output_dir, test_dataset_name, device):
    cfg = get_test_cfg(config_file)
    cfg.defrost()
    cfg.OUTPUT_DIR = detectron_output_dir
    cfg.MODEL.WEIGHTS = os.path.join(detectron_output_dir, "model_final.pth")
    cfg.freeze()
    logger.info(f"Using {test_dataset_name} dataset for evaluating the final model.")
    dataset_dicts = DatasetCatalog.get(test_dataset_name)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    logger.info("Detectron2 model loaded successfully!")

    # target thresholds
    thresholds = np.linspace(0.1, 0.9, 9)

    # Retrieve class names
    metadata = MetadataCatalog.get(test_dataset_name)
    thing_classes = metadata.thing_classes if hasattr(metadata, "thing_classes") else None

    pr_per_class = {}  # Format: {class_name:{"threshold": [...], "Precision": [...], "Recall": [...]}}

    for thresh in thresholds:
        per_class_metrics = evaluate_model_multithresh_per_class(
            model, dataset_dicts, device, score_threshold=thresh, iou_threshold=0.5
        )
        for cid, (p, r) in per_class_metrics.items():
            if thing_classes and cid < len(thing_classes):
                class_name = thing_classes[cid]
            else:
                class_name = f"class_{cid}"
            if class_name not in pr_per_class:
                pr_per_class[class_name] = {"threshold": [], "Precision": [], "Recall": []}
            pr_per_class[class_name]["threshold"].append(float(f"{thresh:.2f}"))
            pr_per_class[class_name]["Precision"].append(float(f"{p:.2f}"))
            pr_per_class[class_name]["Recall"].append(float(f"{r:.2f}"))

    # save to a YAML file.
    yaml_file = os.path.join(detectron_output_dir, "pr_values.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(pr_per_class, f, default_flow_style=False)
    logger.info(f"Per-class PR values saved to {yaml_file}")

    # Log to mlflow.
    for class_name, metrics in tqdm(pr_per_class.items(), desc="Logging per-class metrics"):
        for thresh, p, r in zip(metrics["threshold"], metrics["Precision"], metrics["Recall"]):
            scaled_threshold = int(round(thresh * 10000, 0))
            mlflow.log_metrics({
                f"{class_name}_precision": p,
                f"{class_name}_recall": r
            }, step=scaled_threshold)

    # Overall eval
    thresholds_overall = thresholds
    pr_precisions = []
    pr_recalls = []

    for thresh in thresholds_overall:
        p, r, _, _ = evaluate_model_multithresh(model, dataset_dicts, device,
                                                  score_threshold=thresh, iou_threshold=0.5)
        pr_precisions.append(p)
        pr_recalls.append(r)
        logger.info(f"Threshold: {thresh:.2f}, Precision: {p:.4f}, Recall: {r:.4f}")

    precisions = np.array(pr_precisions)
    recalls = np.array(pr_recalls)
    thresholds_arr = np.array(thresholds_overall)
    target_thresholds = np.linspace(min(thresholds_arr), max(thresholds_arr), 5000)
    interpolated_precisions = np.interp(target_thresholds, thresholds_arr, precisions)
    interpolated_recalls = np.interp(target_thresholds, thresholds_arr, recalls)

    for idx, thresh in tqdm(enumerate(target_thresholds), total=len(target_thresholds), desc="Logging overall metrics"):
        scaled_threshold = int(round(thresh * 10000, 0))
        mlflow.log_metrics({
            'precision': float(interpolated_precisions[idx]),
            'recall': float(interpolated_recalls[idx])
        }, step=scaled_threshold)

    plt.figure(figsize=(8, 6))
    plt.plot(pr_recalls, pr_precisions, marker='o', linestyle='-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid(True)
    pr_curve_path = os.path.join(detectron_output_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path, bbox_inches='tight')
    logger.info(f"Overall PR curve saved to {pr_curve_path}")
