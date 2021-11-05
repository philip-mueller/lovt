import dataclasses
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from models.downstream.downstream_evaluation import ClassificationEvaluationModelConfig
from models.downstream.online_evaluation import OnlineEvaluatorConfig
from models.image.scan_encoder import ScanEncoderConfig
from models.objectives.global_alignment import GlobalNceLossConfig, GlobalPredictorLossConfig
from models.objectives.local_alignment import LocalIntraSampleContrastiveLossConfig, \
    LocalMseLossConfig, LocalPredictorLossConfig
from models.pretraining.pretraining_utils import BiModalModelConfig
from models.text.report_encoder import ReportEncoderConfig

log = logging.getLogger(__name__)


def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
        logger: List[pl.loggers.LightningLoggerBase],
):
    if config.print_config:
        hparams = OmegaConf.to_container(config)
        # save number of model parameters
        params = {}
        params["params_total"] = sum(p.numel() for p in model.parameters())
        params["params_trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        params["params_not_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        hparams['params'] = params

        # send hparams to all loggers
        trainer.logger.log_hyperparams(hparams)

        # disable logging any more hyperparameters for all loggers
        # (this is just to prevent trainer logging hparams of model as we manage it ourselves)
        for lg in logger:
            lg.log_hyperparams = lambda x: None


@dataclass
class BaseExperimentConfig:
    trainer: Dict[str, Any] = dataclasses.field(default_factory=dict)
    callback: Dict[str, Any] = dataclasses.field(default_factory=dict)
    logger: Dict[str, Any] = dataclasses.field(default_factory=dict)

    monitor_metric: str = MISSING
    monitor_metric_mode: str = MISSING

    name: str = '<unnamed>'

    seed: int = 1234

    gpus: Any = -1
    num_dataloader_workers: int = 4

    print_config: bool = True
    debug: bool = False

    work_dir: str = MISSING


def init_trainer(model, trainer_config, gpus, logger_configs, callback_configs, callbacks=None, loggers=None,
                 auto_lr_find=False, terminate_on_nan=True):
    if callbacks is None:
        callbacks = []
    else:
        assert isinstance(callbacks, list)
    for name, cb_conf in callback_configs.items():
        if cb_conf is None:
            continue
        elif "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            raise ValueError(name, cb_conf)
    loggers = init_loggers(logger_configs, loggers)
    log.info(f"Instantiating Trainer ")
    trainer = Trainer(gpus=gpus, max_epochs=model.config.max_epochs,
                      callbacks=callbacks, logger=loggers,
                      num_nodes=1, num_processes=1,
                      terminate_on_nan=not auto_lr_find and terminate_on_nan,
                      auto_lr_find=auto_lr_find,
                      log_every_n_steps=5,
                      **trainer_config)
    #torch.backends.cudnn.benchmark = True
    return loggers, trainer


def init_loggers(logger_configs, loggers=None):
    if loggers is None:
        loggers = []
    else:
        assert isinstance(loggers, list)
    for _, lg_conf in logger_configs.items():
        if lg_conf is None:
            continue
        elif "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
        else:
            raise ValueError(lg_conf)
    return loggers


def setup_training(config):
    if "seed" in config:
        seed_everything(config.seed)
    if config.debug:
        log.info(f"Running in debug mode! <{config.debug=}>")
        config.trainer.fast_dev_run = True
        config.gpus = 0
        config.num_dataloader_workers = 0
        if 'precision' in config.trainer:
            config.trainer.precision = 32
        os.environ['WANDB_MODE'] = 'dryrun'


@dataclass
class DownstreamEvaluatorConfig:
    evaluated_encoder: str = 'a'
    evaluation_model: ClassificationEvaluationModelConfig = MISSING
    callback: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class PreTrainExperimentConfig(BaseExperimentConfig):
    pretrain_dataset: str = MISSING
    pretrain_model: BiModalModelConfig = MISSING
    online_eval: Dict[str, OnlineEvaluatorConfig] = dataclasses.field(default_factory=dict)
    downstream_eval: Dict[str, DownstreamEvaluatorConfig] = dataclasses.field(default_factory=dict)

    monitor_metric: str = 'val/total_loss'  # for checkpoints and early stopping but not for LR scheduler
    monitor_metric_mode: str = 'min'

    auto_lr_find: bool = False


def init_config_store():
    cs = ConfigStore.instance()
    cs.store(name="train_representation_config", node=PreTrainExperimentConfig)
    cs.store(group="pretrain_model", name="base_model", node=BiModalModelConfig)
    cs.store(group="scan_encoder", name="base_scan_encoder", node=ScanEncoderConfig)
    cs.store(group="report_encoder", name="base_report_encoder", node=ReportEncoderConfig)
    cs.store(group="objective", name="base_global_contrastive", node=GlobalNceLossConfig)
    cs.store(group="objective", name="base_global_predictor", node=GlobalPredictorLossConfig)
    cs.store(group="objective", name="local_intra_contrastive", node=LocalIntraSampleContrastiveLossConfig)
    cs.store(group="objective", name="local_mse", node=LocalMseLossConfig)
    cs.store(group="objective", name="base_local_predictor", node=LocalPredictorLossConfig)
