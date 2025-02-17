import os
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    # Compute areas of each box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate_model_multithresh(model, dataset_dicts, device, score_threshold=0.5, iou_threshold=0.5):
    """
    Runs the model on all images in the dataset and computes
    precision, recall, "accuracy" (TP/(TP+FP+FN)) and F1 score.

    Parameters:
      - model: the Detectron2 model.
      - dataset_dicts: list of dicts (from the registered dataset) with keys "file_name" and "annotations".
      - device: torch device to run inference on.
      - score_threshold: minimum score for a prediction to be considered.
      - iou_threshold: IoU threshold to match predictions to ground truth.

    Returns:
      precision, recall, accuracy, and F1 score.
    """
    TP = 0
    FP = 0
    FN = 0

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"Warning: could not read image {d['file_name']}")
            continue
        # BGR to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor_img = torch.as_tensor(img.transpose(2, 0, 1)).to(device)
        inputs = [{"image": tensor_img}]

        with torch.no_grad():
            outputs = model(inputs)

        instances = outputs[0]["instances"].to("cpu")
        if len(instances) == 0:
            pred_boxes = np.array([])
        else:
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]

        # COCO format: [x, y, width, height]
        gt_boxes = []
        for ann in d["annotations"]:
            x1, y1, w, h = ann["bbox"]
            gt_boxes.append([x1, y1, x1 + w, y1 + h])
        gt_boxes = np.array(gt_boxes)

        matched_gt = set()
        for pb in pred_boxes:
            found_match = False
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(pb, gt)
                if iou >= iou_threshold:
                    TP += 1
                    matched_gt.add(i)
                    found_match = True
                    break
            if not found_match:
                FP += 1

        # Ground truth boxes not detected count as false negatives.
        FN += (len(gt_boxes) - len(matched_gt))

    # Compute metrics (guard against division by zero).
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, accuracy, f1


def evaluate_model_multithresh_per_class(model, dataset_dicts, device, score_threshold=0.5, iou_threshold=0.5):
    """
    For each image, computes TP/FP/FN per class and then calculates
    per-class precision and recall.

    Returns:
      A dictionary mapping each class ID to a tuple: (precision, recall)
    """
    per_class_counts = {}

    for d in dataset_dicts:
        for ann in d["annotations"]:
            cid = ann["category_id"]
            if cid not in per_class_counts:
                per_class_counts[cid] = {"TP": 0, "FP": 0, "FN": 0}

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.as_tensor(img.transpose(2, 0, 1)).to(device)
        inputs = [{"image": tensor_img}]

        with torch.no_grad():
            outputs = model(inputs)

        instances = outputs[0]["instances"].to("cpu")
        if len(instances) == 0:
            pred_boxes = np.array([])
            pred_scores = np.array([])
            pred_classes = np.array([])
        else:
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_classes = pred_classes[keep]

        # Group ground truth boxes by class.
        gt_by_class = {}
        for ann in d["annotations"]:
            cid = ann["category_id"]
            bbox = ann["bbox"]  # COCO format: [x, y, width, height]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            if cid not in gt_by_class:
                gt_by_class[cid] = []
            gt_by_class[cid].append(bbox)

        classes_in_img = set(gt_by_class.keys()) | set(pred_classes.tolist())

        for cid in classes_in_img:
            idx = np.where(pred_classes == cid)[0]
            pred_boxes_c = pred_boxes[idx] if len(idx) > 0 else []
            gt_boxes_c = gt_by_class[cid] if cid in gt_by_class else []

            matched = set()
            for pb in pred_boxes_c:
                found_match = False
                for i, gt in enumerate(gt_boxes_c):
                    if i in matched:
                        continue
                    if compute_iou(pb, gt) >= iou_threshold:
                        per_class_counts[cid]["TP"] += 1
                        matched.add(i)
                        found_match = True
                        break
                if not found_match:
                    per_class_counts[cid]["FP"] += 1
            per_class_counts[cid]["FN"] += (len(gt_boxes_c) - len(matched))

    # Compute precision and recall for each class.
    per_class_metrics = {}
    for cid, counts in per_class_counts.items():
        TP = counts["TP"]
        FP = counts["FP"]
        FN = counts["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        per_class_metrics[cid] = (precision, recall)
    return per_class_metrics