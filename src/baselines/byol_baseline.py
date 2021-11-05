import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from byol_pytorch import BYOL
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pixel_level_contrastive_learning import PixelCL
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.insert(0, str(root_path))
    sys.path.append(str(os.path.join(root_path, 'scripts')))


from data.dataloading_utils import load_dataset
from common.wandb import finish_run
from baselines.baseline_utils import load_backbone, BaselineModelConfig, BaselineExperimentConfig, \
    export_pretrained_weights, prepare_transform, DatasetImageOnlyWrapper

from common.script_utils import setup_training, init_trainer, log_hyperparameters

log = logging.getLogger(__name__)


@dataclass
class BYOLConfig(BaselineModelConfig):
    projection_size: int = 128
    ssl_method: str = MISSING


class BYOLModule(pl.LightningModule):
    def __init__(self, config: BYOLConfig, num_workers):
        super().__init__()
        self.save_hyperparameters('config', 'num_workers')
        self.config = config
        self.num_workers = num_workers

        model = load_backbone(config)

        if config.ssl_method in ('BYOL', 'SiamSiam'):
            img_size, transform = prepare_transform(config, mode='SimCLR')

            self.learner = BYOL(model, img_size,
                                hidden_layer='avgpool',
                                projection_size=config.projection_size, projection_hidden_size=4096,
                                augment_fn=transform,
                                use_momentum=config.ssl_method == 'BYOL')
        elif config.ssl_method == 'PixelPro':
            img_size, transform = prepare_transform(config, mode='PixelPro')

            self.learner = PixelCL(model, img_size, hidden_layer_pixel='avgpool', hidden_layer_instance='avgpool',
                                   projection_size=config.projection_size, projection_hidden_size=2048,
                                   augment_fn=transform)
        else:
            raise ValueError(config.ssl_method)

        dataset = load_dataset(self.config.dataset)
        self.train_dataset = dataset['train']

    @property
    def model(self):
        if self.config.ssl_method in ('BYOL', 'SiamSiam'):
            return self.learner.net
        else:
            return self.learner.online_encoder.net

    def training_step(self, x, batch_idx):
        loss = self.learner(x)
        self.log(f'train/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.config.warmup_epochs, max_epochs=self.config.max_epochs
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(DatasetImageOnlyWrapper(self.train_dataset),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def on_before_zero_grad(self, _):
        if isinstance(self.learner, PixelCL) or self.learner.use_momentum:
            self.learner.update_moving_average()


def pretrain_byol_baseline(config: BaselineExperimentConfig):
    setup_training(config)

    log.info(f"----- Initializing Model -----")
    model = BYOLModule(config.model_config, config.num_dataloader_workers)

    log.info(f"----- Initializing Trainer -----")
    logger, trainer = init_trainer(
        model,
        trainer_config=config.trainer,
        gpus=config.gpus,
        logger_configs=config.logger,
        callback_configs=config.callback,
        callbacks=[
            ModelCheckpoint(save_last=True, verbose=True, dirpath='checkpoints/', filename='checkpoint-last')
        ]
    )

    log_hyperparameters(config, model, trainer, logger)

    # Train the model
    log.info(f"----- Starting Training -----")
    trainer.fit(model=model)

    log.info(f"----- Completed training -----")
    finish_run(trainer)
    export_pretrained_weights(config, model, trainer, last_model=True)


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=BaselineExperimentConfig)
cs.store(group="baseline", name="base_model_config", node=BYOLConfig)


@hydra.main(config_path="../../configs/", config_name="baseline")
def main(config: BaselineExperimentConfig) -> None:
    pretrain_byol_baseline(config)


if __name__ == "__main__":
    main()
