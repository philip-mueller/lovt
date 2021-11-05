import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.insert(0, str(root_path))


from baselines.baseline_utils import export_pretrained_weights
from analysis.postprocess_run import update_best_epoch, average_runs
from common.wandb import finish_run
from models.image.scan_encoder import ScanEncoderConfig
from models.text.report_encoder import ReportEncoderConfig
from models.components.utils import EncoderConfig
from models.downstream.downstream_evaluation import ClassificationEvaluationModelConfig, \
    SegmentationEvaluationModelConfig, EvaluationModelConfig, create_downstream_evaluator_for_encoder, \
    DetectionEvaluationModelConfig
from common.script_utils import BaseExperimentConfig, init_trainer, log_hyperparameters, setup_training

log = logging.getLogger(__name__)


@dataclass
class EncoderFinetuningExperimentConfig(BaseExperimentConfig):
    encoder: EncoderConfig = MISSING
    evaluation_model: EvaluationModelConfig = MISSING

    average_runs: int = 1
    export_backbone_weights: bool = False


cs = ConfigStore.instance()
cs.store(name="baseline_fine_tune_config", node=EncoderFinetuningExperimentConfig)
cs.store(group="encoder", name="base_scan_encoder", node=ScanEncoderConfig)
cs.store(group="encoder", name="base_report_encoder", node=ReportEncoderConfig)
cs.store(group="evaluation_model", name="classification_eval", node=ClassificationEvaluationModelConfig)
cs.store(group="evaluation_model", name="segmentation_eval", node=SegmentationEvaluationModelConfig)
cs.store(group="evaluation_model", name="detection_eval", node=DetectionEvaluationModelConfig)


def run_finetuning(config: EncoderFinetuningExperimentConfig):
    setup_training(config)
    run_base_name = config.name
    wandb_runs = []
    for run_i in range(config.average_runs):
        if config.average_runs > 1:
            config.name = f'{run_base_name}[{run_i + 1}]'
        log.info(f"----- Initializing Model -----")
        model = create_downstream_evaluator_for_encoder(
                encoder_config=config.encoder,
                eval_config=config.evaluation_model,
                num_workers=config.num_dataloader_workers
        )

        log.info(f"----- Initializing Trainer -----")
        logger, trainer = init_trainer(
            model,
            trainer_config=config.trainer,
            gpus=config.gpus,
            logger_configs=config.logger,
            callback_configs=config.callback
        )

        log_hyperparameters(config, model, trainer, logger)

        # Train the model
        log.info(f"----- Starting Finetuning -----")
        trainer.fit(model=model)
        log.info(f"----- Completed Finetuning -----")

        log.info(f"----- Testing -----")
        trainer.test()
        log.info(f"----- Done -----")

        if config.export_backbone_weights:
            # segmentation_model.backbone
            export_pretrained_weights(config, model, trainer, last_model=False,
                                      model_extraction_fn=lambda x: x.backbone)

        run_api, dir = finish_run(trainer)
        update_best_epoch(run_api, config.monitor_metric, config.monitor_metric_mode)
        wandb_runs.append(run_api)
    avg_summary = average_runs(wandb_runs)
    for run in wandb_runs:
        run.summary.update(avg_summary)
    return avg_summary


@hydra.main(config_path="../../configs/", config_name="baseline_fine_tune")
def main(config: EncoderFinetuningExperimentConfig):
    return run_finetuning(config)


if __name__ == "__main__":
    main()
