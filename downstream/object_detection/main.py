from detectron2.utils.logger import setup_logger
from downstream.object_detection.config import DetectronConfig
import hydra
from hydra.core.config_store import ConfigStore
import os
import yaml
import pickle
from omegaconf import DictConfig
from scripts.detectron_utils import *
from scripts.train_object_detection import *
from scripts.test_object_detection import *

setup_logger('detectron2_log')

cs = ConfigStore.instance()
cs.store(name="detectron_config", node=DetectronConfig)


@hydra.main(config_path="conf", config_name="dtron_config", version_base=None)
def main(cfg: DetectronConfig):
    train_dataset_name = cfg.data.dataset_name + f"_{cfg.data.label_fraction}perc" + "_train"
    test_dataset_name = cfg.data.dataset_name + "_test"
    config_save_file = cfg.params.final_model_name + f"_{cfg.data.label_fraction}perc" + "_config.pkl"
    output_dir = os.path.join(cfg.paths.output_dir, cfg.params.final_model_name + f"_{cfg.data.label_fraction}perc")

    dtron_config = get_train_cfg(config_file_path=cfg.paths.merge_config_file, pretrained_weights=cfg.paths.pretrained_weights,
                                 train_dataset_name=train_dataset_name, test_dataset_name=test_dataset_name,
                                 num_classes=cfg.data.num_classes, device=cfg.params.device, output_dir=output_dir,
                                 num_workers=cfg.data.num_workers, img_per_batch=cfg.solver.img_per_batch,
                                 base_lr=cfg.solver.base_lr, max_iters=cfg.solver.max_iters,
                                 batch_size_per_image=cfg.model.batch_size_per_image, steps=cfg.solver.steps,
                                 gamma=cfg.solver.gamma, warmup_iters=cfg.solver.warmup_iters,
                                 score_thresh_test=cfg.model.score_thresh_test)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directory {output_dir} created.")

    with open(os.path.join(output_dir, f'{config_save_file}'), 'wb') as f:
        pickle.dump(dtron_config, f, protocol=pickle.HIGHEST_PROTOCOL)

    register_dataset(train_dataset_name, cfg.paths.train_data_path, classes=cfg.data.classes)
    register_dataset(test_dataset_name, cfg.paths.train_data_path, classes=cfg.data.classes)

    logger.info(f"Saving model files to path: {output_dir}")

    if cfg.params.backbone == "resnet50" and cfg.params.trainer == "default":
        logger.info(f"Starting training with Default Trainer")
        train_with_default_trainer(dtron_config, resume=cfg.params.resume)
    if cfg.params.backbone == "resnet50" and cfg.params.trainer == "custom":
        logger.info(f"Starting training with Custom Trainer")
        train_with_custom_trainer(dtron_config, resume=cfg.params.resume)

    load_config_file = os.path.join(output_dir, config_save_file)
    do_coco_eval(load_config_file)


if __name__ == '__main__':
    main()
