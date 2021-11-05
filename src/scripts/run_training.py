import logging
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf
import hydra

if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.insert(0, str(root_path))

from analysis.postprocess_run import update_best_epoch, export_pretrain_run
from common.wandb import finish_run
from models.downstream.downstream_evaluation import ClassificationEvaluator
from common.script_utils import log_hyperparameters, init_trainer, setup_training, PreTrainExperimentConfig, \
    init_config_store
from models.downstream.online_evaluation import instantiate_online_evaluator
from models.pretraining.bimodal_alignment_model import BiModalModelRepresentationLearner

log = logging.getLogger(__name__)

init_config_store()


def run_training(config: PreTrainExperimentConfig):
    setup_training(config)

    log.info(f"----- Initializing Model -----")
    model = BiModalModelRepresentationLearner(OmegaConf.to_container(config.pretrain_model),
                                              dataset=config.pretrain_dataset,
                                              num_workers=config.num_dataloader_workers)

    log.info(f"----- Initializing Trainer -----")
    online_eval_callbacks = []
    for name, oe_conf in config.online_eval.items():
        log.info(f"Instantiating online_evaluator <{oe_conf.task}> for {name}")
        online_eval_callbacks.append(instantiate_online_evaluator(oe_conf, name=name))

    logger, trainer = init_trainer(
        model,
        trainer_config=config.trainer,
        gpus=config.gpus,
        auto_lr_find=config.auto_lr_find,
        logger_configs=config.logger,
        callback_configs=config.callback,
        callbacks=online_eval_callbacks
    )

    log_hyperparameters(config, model, trainer, logger)

    if config.auto_lr_find:
        log.info(f"----- Tuning LR -----")
        trainer.tune(model)
        log.info(f"----- Completed LR Tuning -----")

    # Train the model
    log.info(f"----- Starting Pretraining -----")
    trainer.fit(model=model)

    log.info(f"----- Completed Pretraining -----")

    run_api, _ = finish_run(trainer)
    run_path = os.getcwd()
    os.chdir('..')
    new_path = export_pretrain_run(run_path)
    os.chdir(new_path)


@hydra.main(config_path="../../configs/", config_name="train_representation")
def main(config: PreTrainExperimentConfig):
    return run_training(config)


if __name__ == "__main__":
    main()
