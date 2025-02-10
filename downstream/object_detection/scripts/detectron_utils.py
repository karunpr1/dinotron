import torch
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg, CfgNode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.structures import BoxMode
import random
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import json
import logging
from detectron2.engine import HookBase
import mlflow

# setup_logger('detectron2_log')
logger = logging.getLogger("detectron2_log")


def get_image_dicts(img_dir, json_path):
    """
    Load and parse the COCO annotations JSON file for the given image directory.

    Args:
        img_dir (str): Directory containing the image data
        json_path (str): File path to the COCO annotations JSON file

    Returns:
        list: A list of dictionaries, each representing an image and its annotations.
    """
    with open(json_path) as f:
        coco_dict = json.load(f)

    dataset_dicts = []
    for img_data in coco_dict['images']:
        record = {}

        filename = os.path.join(img_dir, img_data["file_name"])
        height, width = img_data["height"], img_data["width"]

        record["file_name"] = filename
        record["image_id"] = img_data["id"]
        record["height"] = height
        record["width"] = width

        annos = [anno for anno in coco_dict['annotations'] if anno['image_id'] == img_data['id']]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"] - 1,  # Adjust category_id to start from 0
                "segmentation": anno["segmentation"],
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(dataset_name: str, annotation_file:str, dataset_dir: str, classes: list, mode: str):
    """
    Registers a dataset with a given name, directory, and list of classes.

    Args:
        dataset_name (str): The name to register the dataset under.
        dataset_dir (str): The directory where the dataset is stored.
        classes (list): A list of class names corresponding to the dataset.
        annotation_file (str): File path to the COCO annotations JSON file
        mode (str): Mode of registering the dataset to detectron2 framework

    Returns:
        dict: A dictionary containing the registered dataset information.
    """
    if mode == "manual":
        DatasetCatalog.register(dataset_name, lambda: get_image_dicts(dataset_dir, annotation_file))
    else:
        register_coco_instances(dataset_name, {}, annotation_file, dataset_dir)

    MetadataCatalog.get(dataset_name).set(thing_classes=classes)


def plot_samples(dataset_name, n=1):
    """
    Plot a random sample of images from the dataset with annotations.

    Args:
        dataset_name (str): The name of the registered dataset.
        n (int): Number of random samples to plot.
    """
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s['file_name'])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(10, 10))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, pretrained_weights, train_dataset_name, test_dataset_name, num_classes, device,
                  output_dir, num_workers, img_per_batch, base_lr, max_iters, batch_size_per_image, steps, gamma,
                  warmup_iters, score_thresh_test, backbone_freeze_at, mlf_tracking_uri, mlf_exp_name, mlf_run_name,
                  mlf_run_description, test_eval):
    """
    Get the configuration for training the Detectron2 model.

    Args:
        config_file_path (str): Path to the configuration file.
        pretrained_weights (str): URL or path to the pretrained weights.
        train_dataset_name (str): Name of the training dataset.
        test_dataset_name (str): Name of the testing/validation dataset.
        num_classes (int): Number of classes in the dataset.
        device (str): Device to use for training ('cuda' or 'cpu').
        output_dir (str): Directory to save the output model and logs.
        num_workers (int): Number of worker threads used for data loading.
        img_per_batch (int): Number of images processed in each batch during training.
        base_lr (float): Initial learning rate for the optimizer.
        max_iters (int): Maximum number of iterations (batches) to be executed during training.
        batch_size_per_image (int): Number of samples (e.g., region proposals or anchors) used per image during training.
        steps (list): List of iteration indices where the learning rate will be reduced by a factor of `gamma`.
        gamma (float): Factor by which the learning rate is multiplied at each step specified in `steps`.
        warmup_iters (int): Number of iterations for the warmup phase where the learning rate is gradually increased to the base learning rate.
        score_thresh_test (float): Threshold for filtering out low-confidence detections during inference.
        backbone_freeze_at (int): blocks the gradients at the specified convolutional layer
        mlf_tracking_uri (str): Mlfow tracking URI to log metrics
        mlf_exp_name (str): Experiment name to log metrics under mlflow
        mlf_run_name (str): Particular run name to log metrics under mlflow
        mlf_run_description (str): Run description for tracking experiments
        test_eval (int): Evaluation period to model validation during training

    Returns:
        CfgNode: Configuration node with the specified settings.
    """
    cfg = get_cfg()

    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = pretrained_weights
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = num_workers

    cfg.SOLVER.IMS_PER_BATCH = img_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.SOLVER.STEPS = tuple(steps)
    cfg.SOLVER.GAMMA = gamma
    cfg.SOLVER.WARMUP_ITERS = warmup_iters

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze_at
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
    cfg.OUTPUT_DIR = output_dir
    cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION = os.path.join(
        cfg.OUTPUT_DIR, "validation-set-evaluation")
    cfg.OUTPUT_DIR_TEST_SET_EVALUATION = os.path.join(
        cfg.OUTPUT_DIR, "test-set-evaluation")
    cfg.TEST.EVAL_PERIOD = test_eval

    cfg.MLFLOW = CfgNode()
    cfg.MLFLOW.EXPERIMENT_NAME = mlf_exp_name
    cfg.MLFLOW.RUN_DESCRIPTION = mlf_run_description
    cfg.MLFLOW.RUN_NAME = mlf_run_name
    cfg.MLFLOW.TRACKING_URI = mlf_tracking_uri

    cfg.freeze()

    return cfg


def load_checkpoint(filepath):
    """
    Load a PyTorch checkpoint from the specified file path.

    Args:
        filepath (str): The path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint, typically containing the model's state dictionary
              and other metadata.

    Prints:
        The keys available in the loaded checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())
    return checkpoint


def adapt_state_dict(state_dict):
    """
    Adapt the state dictionary by removing specific prefixes and mapping keys
    to the format expected by the Detectron2 model.

    Args:
        state_dict (dict): The original state dictionary from the checkpoint.

    Returns:
        dict: A new state dictionary with adapted keys suitable for loading into
              a Detectron2 model.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove known prefixes
        new_key = k.replace("backbone.", "")
        new_key = new_key.replace("module.", "")

        # Map keys to expected Detectron2 model keys
        new_state_dict[new_key] = v
    return new_state_dict


class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, parameters, and system metrics to MLflow.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        with torch.no_grad():
            mlflow.enable_system_metrics_logging()
            mlflow.set_tracking_uri(self.cfg.MLFLOW.TRACKING_URI)
            mlflow.set_experiment(self.cfg.MLFLOW.EXPERIMENT_NAME)
            mlflow.set_tag("mlflow.note.content", self.cfg.MLFLOW.RUN_DESCRIPTION)
            for k, v in self.cfg.items():
                mlflow.log_param(k, v)

    def after_step(self):
        with torch.no_grad():
            # Log model metrics
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1])

    def after_train(self):
        with torch.no_grad():
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)



class CustomTrainer(DefaultTrainer):
    """
        A custom trainer class that evaluates the model on the validation set every `_C.TEST.EVAL_PERIOD` iterations.
        """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION,
                        exist_ok=True)

        return COCOEvaluator(cfg.DATASETS.TEST[0], distributed=False, output_dir=cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION)


def test_image(dataset_name, predictor, n=2, threshold=0.5):
    """
    Perform inference on a random sample of images from the dataset and visualize the results.

    Args:
        dataset_name (str): The name of the registered dataset.
        predictor (DefaultPredictor): The Detectron2 predictor object for inference.
        n (int): Number of random samples to test.
        threshold (float): Confidence threshold for displaying predictions.
    """
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    for s in random.sample(dataset_custom, n):
        im = cv2.imread(s['file_name'])
        outputs = predictor(im)
        instances = outputs["instances"]
        scores = instances.scores
        keep = scores >= threshold
        instances = instances[keep]
        v = Visualizer(im[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(instances.to("cpu"))

        plt.figure(figsize=(10, 10))
        plt.imshow(v.get_image())
        plt.show()


def coco_evaluator(cfg, predictor, test_dataset_name):
    """
    Evaluate the model using the COCO evaluation metrics on the given test dataset.

    Args:
        cfg (CfgNode): The configuration node containing model and dataset parameters.
        predictor (DefaultPredictor): The Detectron2 predictor object for inference.
        test_dataset_name (str): The name of the registered test dataset.

    Returns:
        dict: The COCO evaluation results.
    """
    evaluator = COCOEvaluator(test_dataset_name, output_dir=cfg.OUTPUT_DIR)
    test_set_loader = build_detection_test_loader(cfg, test_dataset_name)
    results = inference_on_dataset(predictor.model, test_set_loader, evaluator)
    logger.info("Evaluation results for {} in csv format:".format(test_dataset_name))
    logging.info("Evaluation results on test set: %s", results)
    print_csv_format(results)

    return results


def get_test_cfg(config_file_path):
    with open(config_file_path, 'rb') as f:
        cfg = pickle.load(f)
    return cfg

def display_dataset_details(dataset_name):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)

    print(f"Number of samples in {dataset_name}: {len(dataset_dicts)}")
    print(f"{dataset_name} dataset thing_classes: {dataset_metadata.thing_classes}")
