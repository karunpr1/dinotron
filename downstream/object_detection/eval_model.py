import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os
import pickle
from scripts.eval_utils import evaluate_model_multithresh, evaluate_model_multithresh_per_class
from scripts.detectron_utils import *
from config import DetectronConfig
import hydra
from hydra.core.config_store import ConfigStore
import yaml
import copy
import numpy as np

cs = ConfigStore.instance()
cs.store(name="detectron_config", node=DetectronConfig)

@hydra.main(config_path="conf", config_name="dtron_config", version_base=None)
def main(cfg: DetectronConfig):
    test_dataset_name = cfg.evaluate.test_dataset_name
    test_annotations = cfg.evaluate.test_annotations
    test_images = cfg.evaluate.test_images
    device = cfg.params.device

    register_coco_instances(
        test_dataset_name,
        {},
        test_annotations,
        test_images
    )

    metadata = MetadataCatalog.get(test_dataset_name)
    dataset_dicts = DatasetCatalog.get(test_dataset_name)
    thing_classes = metadata.thing_classes if hasattr(metadata, "thing_classes") else None
    logger.info(metadata)
    logger.info(dataset_dicts[0])
    model_output_dir = cfg.evaluate.model_output_dir
    model_name = cfg.evaluate.model_name
    print(model_output_dir)

    pickle_file_path = cfg.evaluate.config_file_path
    with open(pickle_file_path, 'rb') as f:
        cfg = pickle.load(f)
    print(cfg)
    cfg.defrost()
    cfg.OUTPUT_DIR = model_output_dir
    cfg.MODEL.WEIGHTS = os.path.join(model_output_dir, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    predictor = DefaultPredictor(cfg)

    test_output_dir = os.path.join(model_output_dir, "model_evaluation")

    os.makedirs(test_output_dir, exist_ok=True)

    for idx, d in enumerate(random.sample(dataset_dicts, 5)):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        gt_d = copy.deepcopy(d)
        for anno in gt_d.get("annotations", []):
            anno["category_id"] = 0
        gt_metadata = copy.deepcopy(metadata)
        object.__setattr__(gt_metadata, "thing_classes", ["ground truth"])

        visualizer = Visualizer(img[:, :, ::-1], metadata=gt_metadata, scale=0.5)
        vis_gt = visualizer.draw_dataset_dict(gt_d)

        pred_classes = outputs["instances"].pred_classes.cpu().tolist()
        labels = [metadata.thing_classes[i] for i in pred_classes]

        vis_pred = visualizer.overlay_instances(
            labels=labels,
            boxes=outputs["instances"].pred_boxes.tensor.cpu(),
            masks=outputs["instances"].pred_masks.cpu() if outputs["instances"].has("pred_masks") else None,
            assigned_colors=None,
            alpha=0.5,
        )

        # Combine ground truth and predicted visualizations
        combined_image = cv2.addWeighted(
            vis_gt.get_image()[:, :, ::-1], 0.5,
            vis_pred.get_image()[:, :, ::-1], 0.5,
            0
        )

        save_path = os.path.join(test_output_dir, f"{test_dataset_name}_test_image_{idx}.png")
        rotated_image = cv2.rotate(combined_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(save_path, rotated_image)

    print(f"Images saved to {test_output_dir}")

    evaluator = COCOEvaluator(dataset_name=test_dataset_name, output_dir=model_output_dir)
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    print_csv_format(inference)

    # Std eval metrics - overall @ 0.5
    thresh = 0.5
    p, r, acc, f1 = evaluate_model_multithresh(model, dataset_dicts, device,
                                               score_threshold=thresh, iou_threshold=0.5)

    overall_metrics = {
        "Threshold": round(thresh, 2),
        "Precision": round(p, 2),
        "Recall": round(r, 2),
        "Accuracy": round(acc, 2),
        "F1": round(f1, 2)
    }

    logger.info("Standard Evaluation @ 0.5")
    logger.info(overall_metrics)

    # Overall multi-thresh pre-class evaluation
    thresholds = np.linspace(0.1, 0.9, 9)

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


    logger.info("Standard Evaluation per class at multiple threshold")
    logger.info(pr_per_class)

    # overall multi-thresh evaluation
    thresholds_overall = thresholds  # Same thresholds for overall evaluation.
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


    overall_multithresh_metrics = {
        "threshold": [float(round(th, 2)) for th in thresholds_overall],
        "Precision": pr_precisions,
        "Recall": pr_recalls,
        "Accuracy": pr_accuracies,
        "F1": pr_f1
    }

    logger.info("Standard Evaluation overall at multiple threshold")
    logger.info(overall_multithresh_metrics)

    yaml_file = os.path.join(test_output_dir, "test_eval_metrics.yaml")

    # Write COCO evaluation and precision-recall metrics to YAML file
    with open(yaml_file, "w") as f:
        f.write("# <{} test evaluation>\n\n".format(model_output_dir))
        f.write("\n# Standard Evaluation @ 0.5 \n")
        for key, value in overall_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("\n# Standard Evaluation overall at multiple threshold \n")
        for key, value in overall_multithresh_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("\n# Standard Evaluation per class at multiple threshold \n")
        for key, value in pr_per_class.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("# COCO evaluation\n")
        inference_yaml = yaml.dump(dict(inference), default_flow_style=False)
        f.write(inference_yaml)

    print(f"Metrics saved to {yaml_file}")

    # Plot and save the Overall Precision-Recall Curve.
    plt.figure(figsize=(8, 6))
    plt.plot(pr_recalls, pr_precisions, marker='o', linestyle='-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid(True)
    pr_curve_path = os.path.join(test_output_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path, bbox_inches='tight')
    logger.info(f"Overall PR curve saved to {pr_curve_path}")

if __name__ == '__main__':
    main()
