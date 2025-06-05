import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from .detectron_utils import *
from .eval_utils import *
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import yaml
from tqdm import tqdm


def eval_model(config_file, detectron_output_dir, test_dataset_name):
    """
    Evaluates the object detection model using standard evaluation metrics, logs the results to MLflow,
    and plots precision-recall curves for further analysis.

    Params
    config_file (str): Path to the configuration file that defines the model architecture, training parameters, and evaluation settings.
    detectron_output_dir (str) : Directory containing the output of the Detectron2 model (e.g., checkpoints, logs) to be used for evaluation.
    test_dataset_name (str) : Name of the test dataset registered in the DatasetCatalog on which the model evaluation will be performed.

    Returns: None
        The function does not return a value; it logs metrics to MLflow and generates PR curve plots as side effects.
    """

    random.seed(42)
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

    metadata = MetadataCatalog.get(test_dataset_name)
    thing_classes = metadata.thing_classes if hasattr(metadata, "thing_classes") else None

    predictor = DefaultPredictor(cfg)

    test_output_dir = os.path.join(detectron_output_dir, "model_eval")
    os.makedirs(test_output_dir, exist_ok=True)

    for idx, d in enumerate(random.sample(dataset_dicts, 20)):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        visualizer_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis_gt = visualizer_gt.draw_dataset_dict(d)

        visualizer_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        pred_classes = outputs["instances"].pred_classes.cpu().tolist()
        labels = [metadata.thing_classes[i] for i in pred_classes]

        vis_pred = visualizer_pred.overlay_instances(
            labels=labels,
            boxes=outputs["instances"].pred_boxes.tensor.cpu(),
            masks=outputs["instances"].pred_masks.cpu() if outputs["instances"].has("pred_masks") else None,
            assigned_colors=None,
            alpha=0.5,
        )

        # side-by-side compare
        gt_image = vis_gt.get_image()[:, :, ::-1]
        pred_image = vis_pred.get_image()[:, :, ::-1]
        gap_height = int(2 * 96 / 25.4)
        gap = np.zeros((gap_height, gt_image.shape[1], 3), dtype=gt_image.dtype)
        combined_image = np.vstack((gt_image, gap, pred_image))
        rotated_image = cv2.rotate(combined_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        save_path = os.path.join(test_output_dir, f"{test_dataset_name}_test_image_{idx}.png")
        cv2.imwrite(save_path, rotated_image)

    print(f"Images saved to {test_output_dir}")

    evaluator = COCOEvaluator(dataset_name=test_dataset_name, output_dir=detectron_output_dir)
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    print_csv_format(inference)

    if "bbox" in inference:
        for metric_name, metric_value in inference["bbox"].items():
            mlflow.log_metric(
                key=f"COCO/bbox_{metric_name}",
                value=metric_value,
                step=0  # Update step if tracking across epochs
            )

    # Std eval metrics - overall @ 0.5
    prec, rec, TP, FP, FN = compute_precision_recall_fixed_threshold(predictor, dataset_dicts, score_threshold=0.5,
                                                                     iou_threshold=0.5)
    F1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    overall_metrics = {
        "Threshold": 0.5,
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1": round(F1, 2),
        "Accuracy": round(accuracy, 2)
    }

    CM_metrics = {
        "True Positives": TP,
        "False Positives": FP,
        "False Negative": FN
    }

    logger.info("Standard Evaluation @ IoU-0.5")
    logger.info(overall_metrics)
    for key, value in overall_metrics.items():
        mlflow.log_metric(key, value)

    logger.info("CM_metrics @ IoU-0.5")
    logger.info(CM_metrics)
    for key, value in CM_metrics.items():
        mlflow.log_metric(key, value)

    thresholds, recall_array, precision_array = compute_precision_recall_for_thresholds(
        predictor, dataset_dicts, test_output_dir, iou_threshold=0.5, plot=True
    )

    overall_metrics_multithresh = {
        "thresholds": thresholds.tolist() if hasattr(thresholds, "tolist") else list(thresholds),
        "recall": recall_array.tolist() if hasattr(recall_array, "tolist") else list(recall_array),
        "precision": precision_array.tolist() if hasattr(precision_array, "tolist") else list(precision_array)
    }

    per_class_results = compute_precision_recall_for_thresholds_per_class(
        predictor, dataset_dicts, metadata, test_output_dir, iou_threshold=0.5, plot=True, test=True
    )

    per_class_metrics = {}
    for cls, (ths, rec_arr, prec_arr) in per_class_results.items():
        idx = int(cls)
        if metadata is not None and hasattr(metadata, "thing_classes") and idx < len(metadata.thing_classes):
            class_label = metadata.thing_classes[idx]
        else:
            class_label = f"Class {cls}"
        per_class_metrics[class_label] = {
            "thresholds": list(ths),
            "recall": list(rec_arr),
            "precision": list(prec_arr)
        }

    # Log overall metrics to mlflow
    precisions = np.array(precision_array)
    recalls = np.array(recall_array)
    thresholds_arr = np.array(thresholds)

    target_thresholds = np.linspace(min(thresholds_arr), max(thresholds_arr), 5000)
    interpolated_precisions = np.interp(target_thresholds, thresholds_arr, precisions)
    interpolated_recalls = np.interp(target_thresholds, thresholds_arr, recalls)

    for idx, thresh in tqdm(enumerate(target_thresholds), total=len(target_thresholds), desc="Logging overall metrics"):
        scaled_threshold = int(round(thresh * 10000, 0))
        mlflow.log_metrics({
            'precision': float(interpolated_precisions[idx]),
            'recall': float(interpolated_recalls[idx]),
        }, step=scaled_threshold)

    yaml_file = os.path.join(test_output_dir, "test_eval_metrics.yaml")

    with open(yaml_file, "w") as f:
        f.write("# <{} test evaluation>\n\n".format(detectron_output_dir))

        # overall fixed-threshold metrics.
        f.write("Standard Evaluation @ IoU=0.5:\n")
        for key, value in overall_metrics.items():
            f.write(f" {key}: {value}\n")
        f.write("\n")

        # confusion matrix metrics.
        f.write("Confusion Matrix Metrics @ IoU=0.5:\n")
        for key, value in CM_metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # overall evaluation over multiple thresholds.
        f.write("Standard Evaluation Overall (Multiple Score Thresholds):\n")
        for key, value in overall_metrics_multithresh.items():
            if isinstance(value, list):
                formatted_value = "[" + ", ".join(f"{float(v):.2f}" for v in value) + "]"
            f.write(f"{key}: {formatted_value}\n")
        f.write("\n")

        # per-class precision-recall curves.
        f.write("Standard Evaluation per Class (Multiple Score Thresholds):\n")
        for cls, metrics in per_class_metrics.items():
            f.write(f"  Class {cls}:\n")
            thresholds = metrics["thresholds"]
            recall_array = metrics["recall"]
            precision_array = metrics["precision"]
            f.write("    Thresholds: " + ", ".join(f"{float(t):.2f}" for t in thresholds) + "\n")
            f.write("    Recall: " + ", ".join(f"{float(r):.2f}" for r in recall_array) + "\n")
            f.write("    Precision: " + ", ".join(f"{float(p):.2f}" for p in precision_array) + "\n")
        f.write("\n")

        f.write("COCO Evaluation:\n")
        formatted_inference = {k: (v if isinstance(v, (int, float)) else v)
                               for k, v in inference.items()}
        inference_yaml = yaml.dump(formatted_inference, default_flow_style=False)
        f.write(inference_yaml)

    logger.info(f"Metrics saved to {yaml_file}")
    mlflow.log_artifact(yaml_file)


