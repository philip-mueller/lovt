import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC

if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.insert(0, str(root_path))
    sys.path.append(str(os.path.join(root_path, 'scripts')))


from baselines.baseline_utils import load_backbone, BaselineModelConfig, BaselineExperimentConfig, \
    export_pretrained_weights
from analysis.postprocess_run import update_best_epoch
from data.dataloading_utils import load_dataset, DatasetTransformWrapper
from data.datasets.chexpert.chexpert_dataset import chexpert_labels
from models.downstream.classification import MultiTaskBinaryClassifierLoss
from models.image.scan_transforms import ScanAugmentationConfig, ScanDataTransform
from common.config_utils import prepare_config
from common.script_utils import setup_training, init_trainer, log_hyperparameters
from common.wandb import finish_run

log = logging.getLogger(__name__)

"""
python src/baselines/supervised_baseline.py +baseline@model_config=chexpert_resnet50 model_config.dataset=mimic-cxr-img_ap-pa_find-impr name=chexpert-mimic-cxr_ap-pa_find-impr_resnet50
"""


@dataclass
class SupervisedClassificationConfig(BaselineModelConfig):
    classifier_task: str = MISSING
    data_augmentation: ScanAugmentationConfig = ScanAugmentationConfig()

    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    early_stop_patience: int = 10


class SupervisedClassificationModel(pl.LightningModule):
    def __init__(
            self,
            config: SupervisedClassificationConfig, num_workers: int
    ):
        super(SupervisedClassificationModel, self).__init__()

        config: SupervisedClassificationConfig = prepare_config(config, SupervisedClassificationConfig, log)
        self.save_hyperparameters('config', 'num_workers')

        self.config = config
        self.num_workers = num_workers

        dataset_stats = load_dataset(config.dataset)['train'].stats
        if config.classifier_task == 'chexpert_binary':
            classifier_loss = MultiTaskBinaryClassifierLoss(chexpert_labels())
            self.labels_name = 'chexpert_bin_labels'
        elif config.classifier_task == 'chexpert_binary_weighted':
            assert dataset_stats is not None
            classifier_loss = MultiTaskBinaryClassifierLoss(chexpert_labels(),
                                                            pos_weights=dataset_stats.get('chexpert_bin_pos_weights'))
            self.labels_name = 'chexpert_bin_labels'
        else:
            raise ValueError(config.classifier_task)
        num_labels = len(classifier_loss.tasks)

        # Load model and replace fc layer
        self.model = load_backbone(config)
        self.model.fc = nn.Linear(2048, classifier_loss.num_logits, bias=True)
        self.classifier_loss = classifier_loss

        self.train_acc_metric = Accuracy()
        self.val_acc_metric = Accuracy(compute_on_step=False)
        self.test_acc_metric = Accuracy(compute_on_step=False)
        self.val_auroc_metric = AUROC(compute_on_step=False, num_classes=num_labels)
        self.test_auroc_metric = AUROC(compute_on_step=False, num_classes=num_labels)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.transform = ScanDataTransform(config.data_augmentation,
                                           image_size=self.config.input_size, dataset_stats=dataset_stats)
        self.val_transform = ScanDataTransform(config.data_augmentation,
                                               image_size=self.config.input_size, dataset_stats=dataset_stats,
                                               val=True)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, target = batch['scan'], batch['target']
        x = self(x)
        return self.classifier_loss(x, target, return_probs=True)

    def training_step(self, batch, batch_idx):
        probs, labels, loss = self.shared_step(batch)

        acc = self.train_acc_metric(probs, labels)
        self.log(f'train/loss', loss, prog_bar=True)
        self.log(f'train/acc_step', acc, prog_bar=True)
        self.log(f'train/acc_epoch', self.train_acc_metric, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        probs, labels, loss = self.shared_step(batch)
        self.val_acc_metric(probs, labels)
        self.val_auroc_metric(probs, labels)

        self.log(f'val/loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'val/acc', self.val_acc_metric)
        self.log(f'val/auroc', self.val_auroc_metric)

        return loss

    def test_step(self, batch, batch_idx):
        probs, labels, loss = self.shared_step(batch)
        self.test_acc_metric(probs, labels)
        self.test_auroc_metric(probs, labels)

        self.log(f'test/acc', self.test_acc_metric)
        self.log(f'test/auroc', self.test_auroc_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.config.lr_reduce_patience,
                                           factor=self.config.lr_reduce_factor,
                                           mode='max', verbose=True),
            'monitor': 'val/auroc',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            dataset = load_dataset(self.config.dataset)
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
            self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.train_dataset, self.transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.val_dataset, self.val_transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.test_dataset, self.val_transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=True)


def pretrain_supervised_baseline(config: BaselineExperimentConfig):
    setup_training(config)

    log.info(f"----- Initializing Model -----")
    model = SupervisedClassificationModel(config.model_config, config.num_dataloader_workers)

    log.info(f"----- Initializing Trainer -----")
    logger, trainer = init_trainer(
        model,
        trainer_config=config.trainer,
        gpus=config.gpus,
        logger_configs=config.logger,
        callback_configs=config.callback,
        callbacks=[
            ModelCheckpoint(monitor='val/auroc', mode='max', save_top_k=1,
                            save_last=False, verbose=True, dirpath='checkpoints/', filename='checkpoint-{epoch:04d}'),
            EarlyStopping(monitor='val/auroc',  mode='max',
                          patience=config.model_config.early_stop_patience,
                          min_delta=1e-5, verbose=True)
        ]
    )

    log_hyperparameters(config, model, trainer, logger)

    # Train the model
    log.info(f"----- Starting Training -----")
    trainer.fit(model=model)

    log.info(f"----- Completed Training -----")

    log.info(f"----- Testing -----")
    trainer.test()

    run_api, dir = finish_run(trainer)
    update_best_epoch(run_api, 'val/auroc', 'max')
    export_pretrained_weights(config, model, trainer, last_model=False)


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=BaselineExperimentConfig)
cs.store(group="baseline", name="base_model_config", node=SupervisedClassificationConfig)


@hydra.main(config_path="../../configs/", config_name="baseline")
def main(config: BaselineExperimentConfig) -> None:
    pretrain_supervised_baseline(config)


if __name__ == "__main__":
    main()

