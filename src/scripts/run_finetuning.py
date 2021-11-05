import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.insert(0, str(root_path))


from analysis.postprocess_run import update_best_epoch, average_runs, PretrainingRun
from common.wandb import finish_run, get_wandb_logger
from models.downstream.downstream_evaluation import ClassificationEvaluationModelConfig, ClassificationEvaluator, \
    SegmentationEvaluationModelConfig, EvaluationModelConfig, create_downstream_evaluator_for_bimodal_model, \
    DetectionEvaluationModelConfig
from common.script_utils import BaseExperimentConfig, init_trainer, log_hyperparameters, setup_training

log = logging.getLogger(__name__)


@dataclass
class CheckpointFinetuningExperimentConfig(BaseExperimentConfig):
    pretrained_model: str = MISSING
    evaluated_encoder: str = 'a'
    evaluation_model: EvaluationModelConfig = MISSING
    feature_layer: Optional[str] = None

    reported_metrics: List[str] = field(default_factory=list)

    average_runs: int = 1


cs = ConfigStore.instance()
cs.store(name="fine_tune_config", node=CheckpointFinetuningExperimentConfig)
cs.store(group="evaluation_model", name="classification_eval", node=ClassificationEvaluationModelConfig)
cs.store(group="evaluation_model", name="segmentation_eval", node=SegmentationEvaluationModelConfig)
cs.store(group="evaluation_model", name="detection_eval", node=DetectionEvaluationModelConfig)


def run_finetuning(config: CheckpointFinetuningExperimentConfig):
    setup_training(config)
    run_base_name = config.name
    pretrain_run = PretrainingRun.from_run_path(config.pretrained_model)
    wandb_runs = []
    for run_i in range(config.average_runs):
        if config.average_runs > 1:
            config.name = f'{run_base_name}[{run_i+1}]'
        log.info(f"----- Initializing Model -----")
        model, pretrain_config, pretrain_dataset = create_downstream_evaluator_for_bimodal_model(
                checkpoint_path=pretrain_run.best_checkpoint_path,
                encoder=config.evaluated_encoder,
                eval_config=config.evaluation_model,
                num_workers=config.num_dataloader_workers,
                feature_layer=config.feature_layer
            )

        log.info(f"----- Initializing Trainer -----")
        logger, trainer = init_trainer(
            model,
            trainer_config=config.trainer,
            gpus=config.gpus,
            logger_configs=config.logger,
            callback_configs=config.callback
        )

        OmegaConf.set_struct(config, False)
        config.pretrain_model = pretrain_config
        config.pretrain_dataset = pretrain_dataset

        log_hyperparameters(config, model, trainer, logger)

        # Train the model
        log.info(f"----- Starting Finetuning -----")
        trainer.fit(model=model)

        log.info(f"----- Completed Finetuning -----")

        log.info(f"----- Testing -----")
        trainer.test()
        log.info(f"----- Done -----")

        run_api, dir = finish_run(trainer)
        update_best_epoch(run_api, config.monitor_metric, config.monitor_metric_mode)
        wandb_runs.append(run_api)

    avg_summary = average_runs(wandb_runs)
    for run in wandb_runs:
        run.summary.update(avg_summary)
    log.info(f'Reported metrics: {config.reported_metrics}')
    pretrain_run.update_summary(avg_summary, config.evaluation_model.eval_name, include_keys=config.reported_metrics + ['run_ids'])

    return avg_summary


@hydra.main(config_path="../../configs/", config_name="fine_tune")
def main(config: CheckpointFinetuningExperimentConfig):
    return run_finetuning(config)


if __name__ == "__main__":
    main()
