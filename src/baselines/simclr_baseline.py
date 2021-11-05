import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from simclr import SimCLR
from simclr.modules import NT_Xent
from torch.optim import Adam
from torch.utils.data import DataLoader


if __name__ == '__main__':
    baselines_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = baselines_path.parent
    sys.path.insert(0, str(root_path))
    sys.path.append(str(os.path.join(root_path, 'scripts')))

from data.dataloading_utils import load_dataset
from common.script_utils import setup_training, init_trainer, log_hyperparameters
from common.wandb import finish_run
from baselines.baseline_utils import BaselineModelConfig, load_backbone, prepare_transform, \
    TwoImageTransformsWrapper, BaselineExperimentConfig, export_pretrained_weights

log = logging.getLogger(__name__)

@dataclass
class SimCLRConfig(BaselineModelConfig):
    projection_size: int = 128
    temperature: float = 0.1


class SimCLRModule(pl.LightningModule):
    def __init__(self, config: SimCLRConfig, num_workers):
        super().__init__()
        self.save_hyperparameters('config', 'num_workers')
        self.config = config
        self.num_workers = num_workers

        model = load_backbone(config)

        img_size, self.transform = prepare_transform(config, mode='SimCLR')
        self.simclr_model = SimCLR(model, config.projection_size, n_features=model.fc.in_features)
        self.criterion = NT_Xent(config.batch_size, config.temperature, world_size=1)

        dataset = load_dataset(self.config.dataset)
        self.train_dataset = dataset['train']

    @property
    def model(self):
        return self.simclr_model.encoder

    def training_step(self, batch, batch_idx):
        x_i, x_j = batch
        h_i, h_j, z_i, z_j = self.simclr_model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        self.log(f'train/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.config.warmup_epochs, max_epochs=self.config.max_epochs
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(TwoImageTransformsWrapper(self.train_dataset, self.transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)


def pretrain_simclr_baseline(config: BaselineExperimentConfig):
    setup_training(config)

    log.info(f"----- Initializing Model -----")
    model = SimCLRModule(config.model_config, config.num_dataloader_workers)

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
cs.store(group="baseline", name="base_model_config", node=SimCLRConfig)


@hydra.main(config_path="../../configs/", config_name="baseline")
def main(config: BaselineExperimentConfig) -> None:
    pretrain_simclr_baseline(config)


if __name__ == "__main__":
    main()


