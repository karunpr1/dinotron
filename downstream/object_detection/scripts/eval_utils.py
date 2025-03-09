import os
import pickle
import mlflow
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_iou(pred_box, gt_boxes):
    """
    Compute IoU between a single predicted box and an array of ground truth boxes.

    Args:
        pred_box (list or np.ndarray): [x1, y1, x2, y2] for the predicted box.
        gt_boxes (np.ndarray): Array of shape (N, 4) containing GT boxes in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Array of IoU values between pred_box and each of the gt_boxes.
    """
    if len(gt_boxes) == 0:
        return np.array([])

    xA = np.maximum(pred_box[0], gt_boxes[:, 0])
    yA = np.maximum(pred_box[1], gt_boxes[:, 1])
    xB = np.minimum(pred_box[2], gt_boxes[:, 2])
    yB = np.minimum(pred_box[3], gt_boxes[:, 3])
    interW = np.maximum(0, xB - xA)
    interH = np.maximum(0, yB - yA)
    interArea = interW * interH

    boxArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gtAreas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    unionArea = boxArea + gtAreas - interArea
    return interArea / unionArea


def compute_precision_recall_fixed_threshold(predictor, dataset_dicts, score_threshold=0.5, iou_threshold=0.5):
    """
    Compute precision and recall for a Detectron2 model on a test dataset using fixed score and IoU thresholds.

    Args:
        predictor (DefaultPredictor): The Detectron2 predictor (model) used for inference.
        dataset_dicts (dict): test dataset dicts registered in Detectron2's DatasetCatalog.
        score_threshold (float): Confidence score threshold to filter predictions.
        iou_threshold (float): IoU threshold to consider a detection as a true positive.

    Returns:
        tuple: (precision, recall, total_TP, total_FP, total_FN)
    """

    # Initialize counters
    total_TP = 0
    total_FP = 0
    total_FN = 0

    for d in dataset_dicts:
        # Load image
        img = cv2.imread(d["file_name"])
        if img is None:
            continue

        # Run inference
        outputs = predictor(img)
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        valid_idx = np.where(scores >= score_threshold)[0]
        pred_boxes = pred_boxes[valid_idx]
        scores = scores[valid_idx]

        # Convert GT boxes from COCO format ([x, y, w, h]) to [x1, y1, x2, y2]
        gt_boxes = []
        for anno in d.get("annotations", []):
            x, y, w, h = anno["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
        gt_boxes = np.array(gt_boxes)

        # For matching, mark each GT box as not detected initially
        detected = np.zeros(len(gt_boxes), dtype=bool)

        # Process each prediction
        for pred_box in pred_boxes:
            if len(gt_boxes) == 0:
                total_FP += 1
                continue

            ious = compute_iou(pred_box, gt_boxes)
            max_iou = np.max(ious)
            max_idx = np.argmax(ious)

            # If the best IoU exceeds the threshold and the corresponding GT box is not yet matched:
            if max_iou >= iou_threshold and not detected[max_idx]:
                total_TP += 1
                detected[max_idx] = True
            else:
                total_FP += 1

        # Any GT boxes that were not detected are counted as false negatives
        total_FN += np.sum(~detected)

    # Calculate precision and recall
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

    return precision, recall, total_TP, total_FP, total_FN


def compute_precision_recall_for_thresholds(predictor, dataset_dicts, test_output_dir, iou_threshold=0.5, plot=False):
    """
    Compute precision and recall for a Detectron2 model over a test dataset

    Args:
        predictor (DefaultPredictor): The Detectron2 predictor used for inference.
        dataset_dicts (list): Test dataset dictionaries from DatasetCatalog.
        test_output_dir (str): Path of the output save directory
        iou_threshold (float): IoU threshold to consider a detection as a true positive.
        plot (bool): If True, plots the PR curve for each class.

    Returns:
        tuple: (thresholds, recall_array, precision_array) where:
            - thresholds (np.ndarray): The array of score thresholds used.
            - recall_array (np.ndarray): The computed recall for each threshold.
            - precision_array (np.ndarray): The computed precision for each threshold.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    precision_list = []
    recall_list = []

    for score_threshold in thresholds:
        total_TP = 0
        total_FP = 0
        total_FN = 0

        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            if img is None:
                continue

            outputs = predictor(img)
            pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()

            valid_idx = np.where(scores >= score_threshold)[0]
            pred_boxes = pred_boxes[valid_idx]
            # Convert ground truth boxes from COCO format ([x, y, w, h]) to [x1, y1, x2, y2]
            gt_boxes = []
            for anno in d.get("annotations", []):
                x, y, w, h = anno["bbox"]
                gt_boxes.append([x, y, x + w, y + h])
            gt_boxes = np.array(gt_boxes)

            detected = np.zeros(len(gt_boxes), dtype=bool)

            for pred_box in pred_boxes:
                if len(gt_boxes) == 0:
                    total_FP += 1
                    continue

                ious = compute_iou(pred_box, gt_boxes)
                max_iou = np.max(ious)
                max_idx = np.argmax(ious)
                # If the best IoU exceeds the threshold and the corresponding GT box is not yet matched
                if max_iou >= iou_threshold and not detected[max_idx]:
                    total_TP += 1
                    detected[max_idx] = True
                else:
                    total_FP += 1

            total_FN += np.sum(~detected)

        precision = total_TP / (total_TP + total_FP + 1e-10)
        recall = total_TP / (total_TP + total_FN + 1e-10)
        precision_list.append(precision)
        recall_list.append(recall)

    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)

    if plot:
        plt.figure()
        plt.plot(recall_array, precision_array, marker='o')
        for i, th in enumerate(thresholds):
            plt.text(recall_array[i], precision_array[i], f"{th:.1f}", fontsize=8, verticalalignment='bottom')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (IoU = {iou_threshold:.2f})')
        plt.grid(True)
        overall_pr_plot_file = os.path.join(test_output_dir, f"PR_curve_overall@IoU_{iou_threshold:.2f}.png")
        plt.savefig(overall_pr_plot_file)
        mlflow.log_artifact(overall_pr_plot_file)

    return thresholds, recall_array, precision_array


def compute_precision_recall_for_thresholds_per_class(predictor, dataset_dicts, metadata, test_output_dir, iou_threshold=0.5, plot=True):
    """
    Compute precision and recall for each class in the dataset at fixed detection score thresholds.

    Args:
        predictor (DefaultPredictor): The Detectron2 predictor used for inference.
        dataset_dicts (list): Test dataset dictionaries from DatasetCatalog.
        metadata (Metadata): Metadata of the registered detectron test dataset
        test_output_dir (str): Path of the output save directory
        iou_threshold (float): IoU threshold to consider a detection as a true positive.
        plot (bool): If True, plots the PR curve for each class.

    Returns:
        dict: A dictionary mapping each class ID to a tuple (thresholds, recall_array, precision_array),
              where thresholds is an array of score thresholds, and recall_array and precision_array
              are computed for each threshold.
    """
    thresholds_arr = np.linspace(0.1, 0.9, 9)
    image_results = []
    classes_set = set()

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            continue

        outputs = predictor(img)
        # Extract predicted boxes, scores, and classes (pred_classes is a tensor of class indices)
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy()
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()

        # Expecting each annotation to have "bbox" (COCO format: [x, y, w, h]) and "category_id".
        gt_by_class = {}
        for anno in d.get("annotations", []):
            cls = anno["category_id"]
            classes_set.add(cls)
            x, y, w, h = anno["bbox"]
            box = [x, y, x + w, y + h]
            if cls not in gt_by_class:
                gt_by_class[cls] = []
            gt_by_class[cls].append(box)
        for cls in gt_by_class:
            gt_by_class[cls] = np.array(gt_by_class[cls])

        pred_by_class = {}
        for score, box, cls in zip(pred_scores, pred_boxes, pred_classes):
            classes_set.add(cls)
            if cls not in pred_by_class:
                pred_by_class[cls] = []
            pred_by_class[cls].append((score, box))
        for cls in pred_by_class:
            pred_by_class[cls].sort(key=lambda x: x[0], reverse=True)

        image_results.append({
            'gt': gt_by_class,
            'pred': pred_by_class
        })

    results = {}  # key: class, value: (thresholds, recall_array, precision_array)
    for cls in sorted(classes_set):
        precision_list = []
        recall_list = []
        for thresh in thresholds_arr:
            total_TP = 0
            total_FP = 0
            total_FN = 0
            for res in image_results:
                gt_boxes = res['gt'].get(cls, np.empty((0, 4)))
                pred_list = res['pred'].get(cls, [])
                pred_filtered = [box for (score, box) in pred_list if score >= thresh]
                detected = np.zeros(len(gt_boxes), dtype=bool)

                for pred_box in pred_filtered:
                    if len(gt_boxes) == 0:
                        total_FP += 1
                        continue
                    ious = compute_iou(pred_box, gt_boxes)
                    max_iou = np.max(ious) if ious.size > 0 else 0
                    max_idx = np.argmax(ious) if ious.size > 0 else -1

                    if max_iou >= iou_threshold and not detected[max_idx]:
                        total_TP += 1
                        detected[max_idx] = True
                    else:
                        total_FP += 1
                # Ground truth boxes not detected are false negatives.
                total_FN += np.sum(~detected)
            prec = total_TP / (total_TP + total_FP + 1e-10) if (total_TP + total_FP) > 0 else 0.0
            rec = total_TP / (total_TP + total_FN + 1e-10) if (total_TP + total_FN) > 0 else 0.0
            precision_list.append(prec)
            recall_list.append(rec)
        precision_array = np.array(precision_list)
        recall_array = np.array(recall_list)
        results[cls] = (thresholds_arr, recall_array, precision_array)

    # Plot the Precision-Recall curve
    if plot:
        plt.figure()
        cmap = plt.get_cmap("tab10")
        sorted_classes = sorted(results.keys())
        for i, cls in enumerate(sorted_classes):
            ths, rec_arr, prec_arr = results[cls]
            color = cmap(i % 10)
            if metadata is not None and hasattr(metadata, "thing_classes") and cls < len(metadata.thing_classes):
                label = f"{metadata.thing_classes[cls]}"
            else:
                label = f"Class {cls}"
            plt.plot(rec_arr, prec_arr, marker='o', color=color, label=label)
            for j, th in enumerate(ths):
                plt.text(rec_arr[j], prec_arr[j], f"{th:.1f}", fontsize=8, verticalalignment='bottom', color=color)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve per Class (IoU = {iou_threshold:.2f})')
        plt.legend(loc='upper right')
        plt.grid(True)
        per_class_pr_plot_file = os.path.join(test_output_dir, f"PR_curve_per_class_IoU_{iou_threshold:.2f}.png")
        plt.savefig(per_class_pr_plot_file)
        mlflow.log_artifact(per_class_pr_plot_file)

    return results
