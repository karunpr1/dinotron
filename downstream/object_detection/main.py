from config import DetectronConfig
import hydra
from hydra.core.config_store import ConfigStore
from scripts.train_object_detection import *
from scripts.test_object_detection import *

setup_logger('detectron2_log')

cs = ConfigStore.instance()
cs.store(name="detectron_config", node=DetectronConfig)


@hydra.main(config_path="conf", config_name="dtron_config", version_base=None)
def main(cfg: DetectronConfig):
    detectron_output_dir = os.path.join("./detectron_output", cfg.paths.output_dir)
    if not os.path.exists(detectron_output_dir):
        os.makedirs(detectron_output_dir)
        logger.info(f"Directory {cfg.paths.output_dir} created under {detectron_output_dir}.")

    train_dataset_name = cfg.data.dataset_name + f"_{cfg.data.label_fraction}perc" + "_train"
    test_dataset_name = cfg.data.dataset_name + "_test"
    config_save_file = cfg.params.final_model_name + "_config.pkl"

    dtron_config = get_train_cfg(config_file_path=cfg.paths.merge_config_file, pretrained_weights=cfg.paths.pretrained_weights,
                                 train_dataset_name=train_dataset_name, test_dataset_name=test_dataset_name,
                                 num_classes=cfg.data.num_classes, device=cfg.params.device, output_dir=detectron_output_dir,
                                 num_workers=cfg.data.num_workers, img_per_batch=cfg.solver.img_per_batch,
                                 base_lr=cfg.solver.base_lr, max_iters=cfg.solver.max_iters,
                                 batch_size_per_image=cfg.model.batch_size_per_image, steps=cfg.solver.steps,
                                 gamma=cfg.solver.gamma, warmup_iters=cfg.solver.warmup_iters,
                                 score_thresh_test=cfg.model.score_thresh_test,
                                 backbone_freeze_at=cfg.model.backbone_freeze_at)



    with open(os.path.join(detectron_output_dir, f'{config_save_file}'), 'wb') as f:
        pickle.dump(dtron_config, f, protocol=pickle.HIGHEST_PROTOCOL)

    register_dataset(dataset_name=train_dataset_name, annotation_file=cfg.paths.train_annotations_file,
                     dataset_dir=cfg.paths.train_image_path, classes=cfg.data.classes, mode=cfg.data.register_dataset)
    register_dataset(dataset_name=test_dataset_name, annotation_file=cfg.paths.test_annotations_file,
                     dataset_dir=cfg.paths.test_image_path, classes=cfg.data.classes, mode=cfg.data.register_dataset)
    display_dataset_details(train_dataset_name)
    display_dataset_details(test_dataset_name)

    logger.info(f"Saving model files to path: {detectron_output_dir}")

    if cfg.params.backbone == "resnet50" and cfg.params.trainer == "default":
        logger.info(f"Starting training with Default Trainer")
        train_with_default_trainer(dtron_config, resume=cfg.params.resume)
    if cfg.params.backbone == "resnet50" and cfg.params.trainer == "custom":
        logger.info(f"Starting training with Custom Trainer")
        train_with_custom_trainer(dtron_config, resume=cfg.params.resume)

    logger.info(f"Training Completed")
    logger.info(f"Loading files for evaluation.....")
    load_config_file = os.path.join(detectron_output_dir, config_save_file)
    do_coco_eval(load_config_file, detectron_output_dir)


if __name__ == '__main__':
    main()
