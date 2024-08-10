import torch
from detectron2.utils.logger import setup_logger
setup_logger('detectron2')
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.structures import BoxMode


import random
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import json


def get_image_dicts(img_dir):
    """
    Load and parse the COCO annotations JSON file for the given image directory.

    Args:
        img_dir (str): Directory containing the image data and COCO annotations JSON file.

    Returns:
        list: A list of dictionaries, each representing an image and its annotations.
    """
    json_file = os.path.join(img_dir, "_annotations.coco.json")
    with open(json_file) as f:
        coco_dict = json.load(f)

    dataset_dicts = []
    for img_data in coco_dict['images']:
        record = {}

        filename = os.path.join(img_dir, 'images', img_data["file_name"])
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


def register_dataset(dataset_name: str, dataset_dir: str, classes: list):
    """
    Registers a dataset with a given name, directory, and list of classes.

    Args:
        dataset_name (str): The name to register the dataset under.
        dataset_dir (str): The directory where the dataset is stored.
        classes (list): A list of class names corresponding to the dataset.

    Returns:
        dict: A dictionary containing the registered dataset information.
    """
    DatasetCatalog.register(dataset_name, lambda: get_image_dicts(dataset_dir))
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


def get_train_cfg(config_file_path, pretrained_weights, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
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

    Returns:
        CfgNode: Configuration node with the specified settings.
    """
    cfg = get_cfg()

    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = pretrained_weights
    cfg.DATASETS.TRAIN = (train_dataset_name, )
    cfg.DATASETS.TEST = (test_dataset_name, )

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 18000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.SOLVER.STEPS = (12000, 16000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 3000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = output_dir

    return cfg


def get_test_cfg(config_file_path):
    # TODO: Add logic to get cfg from the pickle file
    with open(config_file_path, 'rb') as f:
        cfg = pickle.load(f)


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


# Custom class to load pre-trained DINO ResNet50 backbone (student network)
class CustomTrainer(DefaultTrainer):
    """
    Custom trainer class that extends the Detectron2 DefaultTrainer to load a
    pre-trained DINO ResNet50 backbone (student network) for model training.
    """
    @classmethod
    def build_model(cls, cfg):
        """
        Build the model based on the provided configuration.

        Args:
            cfg (CfgNode): The configuration object for the model.

        Returns:
            nn.Module: The constructed model.
        """
        model = super().build_model(cfg)
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build the data loader for training.

        Args:
            cfg (CfgNode): The configuration object for the data loader.

        Returns:
            DataLoader: The data loader for training.
        """
        return build_detection_train_loader(cfg)

    def resume_or_load(self, resume=True):
        """
        Resume training from a checkpoint or load a pre-trained model.

        Args:
            resume (bool): If True, resumes training from the last checkpoint. If False or
                           if no checkpoint exists, loads the model from the pre-trained weights
                           specified in the configuration.

        Raises:
            KeyError: If the checkpoint does not contain a 'student' key.

        Behavior:
            - If a checkpoint exists in the output directory and `resume` is True, resumes
              training from the checkpoint.
            - Otherwise, loads the specified checkpoint, extracts the 'student' weights, adapts
              the state dictionary, and loads it into the model.
        """
        if resume and os.path.isfile(os.path.join(self.cfg.OUTPUT_DIR, "last_checkpoint")):
            super().resume_or_load(resume)
        else:
            # Load the checkpoint
            checkpoint = load_checkpoint(self.cfg.MODEL.WEIGHTS)

            # Extract the 'student' weights
            if 'student' in checkpoint:
                state_dict = checkpoint['student']
            else:
                raise KeyError("The checkpoint does not contain a 'student' key.")

            adapted_state_dict = adapt_state_dict(state_dict)
            self.model.load_state_dict(adapted_state_dict, strict=False)


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
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

