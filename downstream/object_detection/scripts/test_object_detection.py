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
    """
    Evaluates the object detection model using standard evaluation metrics, logs the results to MLflow,
    and plots precision-recall curves for further analysis.

    Params
    config_file (str): Path to the configuration file that defines the model architecture, training parameters, and evaluation settings.
    detectron_output_dir (str) : Directory containing the output of the Detectron2 model (e.g., checkpoints, logs) to be used for evaluation.
    test_dataset_name (str) : Name of the test dataset registered in the DatasetCatalog on which the model evaluation will be performed.
    device (str) : Device identifier for running the evaluation (e.g., "cpu", "cuda").

    Returns: None
        The function does not return a value; it logs metrics to MLflow and generates PR curve plots as side effects.
    """

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

    yaml_file = os.path.join(detectron_output_dir, "pr_values.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(pr_per_class, f, default_flow_style=False)
    logger.info(f"Per-class PR values saved to {yaml_file}")

    # Log to mlflow.
    for class_name, metrics in pr_per_class.items():
        for thresh, p, r in zip(metrics["threshold"], metrics["Precision"], metrics["Recall"]):
            scaled_threshold = int(round(thresh * 10000, 0))
            mlflow.log_metrics({
                f"{class_name}_precision": p,
                f"{class_name}_recall": r
            }, step=scaled_threshold)

    # Overall evaluation
    thresholds_overall = thresholds
    pr_precisions = []
    pr_recalls = []
    pr_accuracies = []
    pr_f1 = []

    for thresh in thresholds_overall:
        p, r, acc, f1 = evaluate_model_multithresh(model, dataset_dicts, device,
                                                   score_threshold=thresh, iou_threshold=0.5)
        pr_precisions.append(round(p, 2))
        pr_recalls.append(round(r, 2))
        pr_accuracies.append(round(acc, 2))
        pr_f1.append(round(f1, 2))
        logger.info(
            f"Overall - Threshold: {thresh:.2f}, Precision: {p:.4f}, Recall: {r:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")

    overall_metrics = {
        "threshold": [float(round(th, 2)) for th in thresholds_overall],
        "Precision": pr_precisions,
        "Recall": pr_recalls,
        "Accuracy": pr_accuracies,
        "F1": pr_f1
    }

    yaml_file = os.path.join(detectron_output_dir, "model_performance_metrics.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(overall_metrics, f, default_flow_style=False)

    logger.info(f"Overall model performance metrics saved to {yaml_file}")

    mlflow.log_dict(overall_metrics, "model_performance_metrics.yaml")

    # Convert lists to numpy arrays.
    precisions = np.array(pr_precisions)
    recalls = np.array(pr_recalls)
    accuracies = np.array(pr_accuracies)
    f1s = np.array(pr_f1)
    thresholds_arr = np.array(thresholds_overall)

    # Interpolate for finer threshold logging.
    target_thresholds = np.linspace(min(thresholds_arr), max(thresholds_arr), 5000)
    interpolated_precisions = np.interp(target_thresholds, thresholds_arr, precisions)
    interpolated_recalls = np.interp(target_thresholds, thresholds_arr, recalls)
    interpolated_accuracies = np.interp(target_thresholds, thresholds_arr, accuracies)
    interpolated_f1 = np.interp(target_thresholds, thresholds_arr, f1s)

    # Log overall metrics to mlflow using tqdm for progress.
    from tqdm import tqdm
    for idx, thresh in tqdm(enumerate(target_thresholds), total=len(target_thresholds), desc="Logging overall metrics"):
        scaled_threshold = int(round(thresh * 10000, 0))
        mlflow.log_metrics({
            'precision': float(interpolated_precisions[idx]),
            'recall': float(interpolated_recalls[idx]),
            'accuracy': float(interpolated_accuracies[idx]),
            'f1': float(interpolated_f1[idx])
        }, step=scaled_threshold)

    # Plot and save the Overall Precision-Recall Curve.
    plt.figure(figsize=(8, 6))
    plt.plot(pr_recalls, pr_precisions, marker='o', linestyle='-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid(True)
    pr_curve_path = os.path.join(detectron_output_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path, bbox_inches='tight')
    logger.info(f"Overall PR curve saved to {pr_curve_path}")

    # Plot and save the F1 Score vs. Threshold Curve.
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_overall, pr_f1, linestyle='-')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Threshold")
    plt.grid(True)
    f1_curve_path = os.path.join(detectron_output_dir, 'f1_curve.png')
    plt.savefig(f1_curve_path, bbox_inches='tight')
    logger.info(f"F1 score curve saved to {f1_curve_path}")

