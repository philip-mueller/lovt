import logging
import os
import glob
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator

import click
import wandb
import wandb.wandb_run
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.experimental import initialize, compose
from pytorch_lightning import Trainer
from wandb.apis.public import Run

if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.append(str(root_path))

log = logging.getLogger(__name__)


from models.image.scan_encoder import ScanEncoderConfig
from models.pretraining.bimodal_alignment_model import BiModalModelRepresentationLearner
from analysis.downstream_embeddings import LocalScanDownstreamEmbeddingsExporter, LocalScanDownstreamEmbeddings
from models.pretraining.pretraining_utils import EmbeddingsData, ModelInputData, AttentionData
from common.script_utils import init_config_store
from common.dataclass_utils import TensorDataclassMixin
from common.user_config import USER_CONFIG


def load_wandb_run_from_id(run_id) -> Run:
    return wandb.Api().run(f"{USER_CONFIG.wandb.user}/{USER_CONFIG.wandb.project}/{run_id}")


def get_run_id_from_path(run_path):
    assert os.path.exists(run_path), f'Run not found at {run_path}'
    wandb_folder = os.path.join(run_path, 'wandb')
    wandb_files = glob.glob(f'{wandb_folder}/run-*')
    assert len(wandb_files) == 1, wandb_files
    run_file = wandb_files[0]
    return run_file[-8:]


def export_pretrain_run(run_path):
    log.info(f'Exporting pretrain run from path {run_path}')
    run = load_wandb_run_from_id(get_run_id_from_path(run_path))
    target_path = os.path.join(USER_CONFIG.models.base_path, USER_CONFIG.models.pretrained_models_folder, run.name)
    assert not os.path.exists(target_path)
    shutil.move(run_path, target_path)
    log.info(f'Moved run to {target_path}')
    update_run(target_path)
    return target_path


def update_run(run_path):
    run_path = os.path.abspath(run_path)
    run = load_wandb_run_from_id(get_run_id_from_path(run_path))
    rel_path = os.path.relpath(run_path, USER_CONFIG.models.base_path)
    run.config['run_path'] = rel_path
    log.info(f'Set run path to {rel_path}')
    run = PretrainingRun(run, run_path)
    run.compute_best_epoch()
    if not os.path.exists(run.best_checkpoint_path):
        shutil.move(os.path.join(run.run_path, 'checkpoints', f'pretrain-epoch={run.best_epoch:04d}.ckpt'),
                    run.best_checkpoint_path)


class PretrainingRun:
    def __init__(self, run, run_path):
        self.run: Run = run
        self.run_path = run_path
        assert os.path.exists(run_path), run_path

    @staticmethod
    def from_run_id(run_id) -> 'PretrainingRun':
        run = load_wandb_run_from_id(run_id)
        run_path = os.path.join(USER_CONFIG.models.base_path, run.config['run_path'])
        return PretrainingRun(run, run_path)

    @staticmethod
    def from_run_path(pretrain_run_name) -> 'PretrainingRun':
        run_path = os.path.join(USER_CONFIG.models.base_path, USER_CONFIG.models.pretrained_models_folder, pretrain_run_name)
        run = load_wandb_run_from_id(get_run_id_from_path(run_path))
        return PretrainingRun(run, run_path)

    @staticmethod
    def from_pretrain_run_info(pretrain_run_info) -> 'PretrainingRun':
        run = wandb.api.run(pretrain_run_info['wandb_path'])
        run_path = os.path.join(USER_CONFIG.models.base_path, run.config['run_path'])
        return PretrainingRun(run, run_path)

    @staticmethod
    def from_analysis_run(analysis_run: Run) -> 'PretrainingRun':
        run = wandb.api.run(analysis_run.config['pretrain_run/wandb_path'])
        run_path = os.path.join(USER_CONFIG.models.base_path, run.config['run_path'])
        return PretrainingRun(run, run_path)

    @property
    def pretrain_run_info(self):
        return {
            'name': self.run.name,
            'wandb_path': self.run.path,
            'epoch': self.best_epoch
        }

    def link_with_analysis_run(self, analysis_run: Run):
        analysis_run.config.update({'pretrain_run/' + key: value for key, value in self.pretrain_run_info.items()})
        analysis_run.update()

    def compute_best_epoch(self):
        update_best_epoch(self.run, self.run.config.get('monitor_metric', 'val/total_loss'), self.run.config.get('monitor_metric_mode', 'min'))

    @property
    def best_epoch(self) -> int:
        return int(self.run.summary['epoch'])

    @property
    def best_checkpoint_path(self):
        path = os.path.join(self.run_path, 'checkpoints', f'pretrain-epoch{self.best_epoch:04d}.ckpt')
        return path

    def load_best_model(self, do_load_dataset=True) -> 'BiModalModelRepresentationLearner':
        assert os.path.exists(self.best_checkpoint_path), f'checkpoint not found at path {self.best_checkpoint_path}'
        from models.pretraining.bimodal_alignment_model import BiModalModelRepresentationLearner
        return BiModalModelRepresentationLearner.load_from_checkpoint(self.best_checkpoint_path, strict=True,
                                                                      do_load_dataset=do_load_dataset)

    def update_summary(self, summary_dict, analysis_name, analysis_run=None, include_keys=None):
        if include_keys is not None and len(include_keys) > 0:
            print('Found keys: ', summary_dict.keys())
            summary_dict = {key: value for key, value in summary_dict.items() if key in include_keys}
        else:
            summary_dict = dict(summary_dict)
        if analysis_run is not None:
            summary_dict['run/name'] = analysis_run.name
            summary_dict['run/wandb_path'] = analysis_run.cache_path

        self.run.summary.update({f'{analysis_name}/{key}': value for key, value in summary_dict.items()})
        self.run.update()

    def compute_model_data(self, gpus, data='test'):
        model = self.load_best_model()
        model.run_dir = self.run_path
        model.predictions_sub_folder = data
        model.setup()
        trainer = Trainer(gpus=gpus)
        trainer.predict(model=model, dataloaders=model.get_dataloader(data))

    def load_model_data(self, batch_idx: int, data='test',
                        load_inputs=True, load_embeddings=True, load_attentions=True, device='cpu') -> 'ModelData':
        predictions_dir = os.path.join(self.run_path, 'predictions', data)
        assert self.num_model_data_batches(data) > 0
        model_data = ModelData()
        if load_inputs:
            model_data.inputs = ModelInputData.load(predictions_dir, batch_idx, device=device)
        if load_embeddings:
            model_data.embeddings = EmbeddingsData.load(predictions_dir, batch_idx, device=device)
        if load_attentions:
            model_data.attentions = AttentionData.load(predictions_dir, batch_idx, device=device)
        return model_data

    def num_model_data_batches(self, data) -> int:
        predictions_dir = os.path.join(self.run_path, 'predictions', data)
        num_batches = min(ModelInputData.num_batches(predictions_dir),
                   EmbeddingsData.num_batches(predictions_dir),
                   AttentionData.num_batches(predictions_dir))
        assert num_batches > 0, f'No model data found for run: {predictions_dir}'
        return num_batches

    def iter_model_data_batches(self, data='test',
                                load_inputs=True, load_embeddings=True, load_attentions=True) -> Iterator['ModelData']:
        for idx in range(self.num_model_data_batches(data)):
            yield self.load_model_data(idx, data=data,
                                              load_inputs=load_inputs,
                                              load_embeddings=load_embeddings,
                                              load_attentions=load_attentions)

    def compute_downstream_data(self, gpu, task, dataset, data='test'):
        model = self.load_best_model()
        predictions_dir = self.downstream_predictions_dir(data, dataset)
        LocalScanDownstreamEmbeddingsExporter.export_model(model, predictions_dir,
                                                           task,
                                                           dataset=dataset, data=data, gpu=gpu)

    def downstream_predictions_dir(self, data, dataset):
        return os.path.join(self.run_path, 'predictions', f'downstream_{dataset}_{data}')

    def load_downstream_data(self, batch_idx: int, dataset, data='test', device='cpu') -> LocalScanDownstreamEmbeddings:
        predictions_dir = self.downstream_predictions_dir(data, dataset)
        return LocalScanDownstreamEmbeddings.load(predictions_dir, batch_idx, device=device)

    def num_downstream_data_batches(self, dataset, data):
        predictions_dir = self.downstream_predictions_dir(data, dataset)
        return LocalScanDownstreamEmbeddings.num_batches(predictions_dir)

    def iter_downstream_data_batches(self, dataset, data='test') -> Iterator[LocalScanDownstreamEmbeddings]:
        for idx in range(self.num_downstream_data_batches(dataset, data)):
            yield self.load_downstream_data(idx, dataset, data=data)


@dataclass
class ModelData(TensorDataclassMixin):
    inputs: Optional[ModelInputData] = None
    embeddings: Optional[EmbeddingsData] = None
    attentions: Optional[AttentionData] = None

    @property
    def num_samples(self):
        return self.embeddings.yl_a.shape[0]


def update_best_epoch(run, monitor_metric, monitor_metric_mode='max'):
    assert isinstance(run, wandb.apis.public.Run)
    all_metrics = [key for key in run.summary.keys() if 'val' in key]
    assert monitor_metric in all_metrics, monitor_metric

    epoch_val_results = [epoch_result for epoch_result in run.scan_history() if monitor_metric in epoch_result]
    best_epoch_metrics = [epoch_result[monitor_metric] for epoch_result in epoch_val_results]

    if monitor_metric_mode == 'max':
        best_index = np.argmax(best_epoch_metrics)
    elif monitor_metric_mode == 'min':
        best_index = np.argmin(best_epoch_metrics)
    else:
        raise ValueError(monitor_metric_mode)

    best_epoch = epoch_val_results[best_index]

    for key, value in best_epoch.items():
        if key in ["epoch"] + all_metrics:
            run.summary[key] = value
    run.update()
    log.info(f'Set best epoch to {best_index}')


def average_runs(runs: list) -> dict:
    run_ids = ';'.join(run.id for run in runs)
    summaries = defaultdict(list)
    for run in runs:
        for key, value in run.summary.items():
            if isinstance(value, (float, int)):
                summaries[key].append(value)

    avg_summary = {'run_ids': run_ids}
    for key, values in summaries.items():
        values = np.array(values, dtype=float)
        avg_summary[f'{key}__runs_mean'] = values.mean()
        avg_summary[f'{key}__runs_std'] = values.std()
    return avg_summary


def compute_untrained_model_data(model_config, run_dir, gpus, dataset='mimic-cxr_all_find-impr', data='test'):
    model = BiModalModelRepresentationLearner(model_config, dataset=dataset)
    model.run_dir = run_dir
    model.predictions_sub_folder = data
    model.setup()
    trainer = Trainer(gpus=gpus)
    trainer.predict(model=model, dataloaders=model.get_dataloader(data))


class BaselineRun:
    def __init__(self, baseline_name):
        self.baseline_name = baseline_name
        self.run_path = os.path.join(USER_CONFIG.models.base_path, 'baselines', baseline_name)

        if baseline_name in ('random', 'ImageNet'):
            os.makedirs(self.run_path, exist_ok=True)
            self.has_run = False
        else:
            assert os.path.exists(self.run_path), self.run_path
            self.has_run = True

    @property
    def run(self):
        if self.has_run:
            return load_wandb_run_from_id(get_run_id_from_path(self.run_path))
        else:
            return None

    @property
    def backbone_checkpoint(self):
        if self.has_run:
            return os.path.join(self.run_path, 'checkpoints', 'backbone_weights.pt')
        else:
            return None

    @property
    def encoder_config(self):
        # this is the default encoder config to evaluate baseline runs that only have a backbone but no whole encoder
        return ScanEncoderConfig(
            backbone_architecture='resnet',
            backbone_model=('pytorch/vision:v0.6.0', 'resnet50'),
            backbone_checkpoint=f'baseline:{self.baseline_name}' if self.has_run else None,
            backbone_pretrained=self.baseline_name == 'ImageNet',
            input_size=(224, 224),
            region_feature_layer='conv5',
            global_feature_layer='conv5',
            global_aggregator='avg'
        )

    def compute_downstream_data(self, gpu, task, dataset, data='test'):
        predictions_dir = self.downstream_predictions_dir(data, dataset)
        LocalScanDownstreamEmbeddingsExporter.export_baseline(self.encoder_config, predictions_dir,
                                                              task,
                                                              dataset=dataset, data=data, gpu=gpu)

    def downstream_predictions_dir(self, data, dataset):
        return os.path.join(self.run_path, 'predictions', f'downstream_{dataset}_{data}')

    def load_downstream_data(self, batch_idx: int, dataset, data='test', device='cpu') -> LocalScanDownstreamEmbeddings:
        predictions_dir = self.downstream_predictions_dir(data, dataset)
        return LocalScanDownstreamEmbeddings.load(predictions_dir, batch_idx, device=device)

    def num_downstream_data_batches(self, dataset, data):
        predictions_dir = self.downstream_predictions_dir(data, dataset)
        return LocalScanDownstreamEmbeddings.num_batches(predictions_dir)

    def iter_downstream_data_batches(self, dataset, data='test') -> Iterator[LocalScanDownstreamEmbeddings]:
        for idx in range(self.num_downstream_data_batches(dataset, data)):
            yield self.load_downstream_data(idx, dataset, data=data)


@click.group()
def cli():
    pass


@cli.command('export_pretrain_run')
@click.argument('run_path')
def export_pretrain_run_cmd(run_path):
    export_pretrain_run(run_path)


@cli.command('update_pretrain_run')
@click.argument('run_path')
def update_run_cmd(run_path):
    update_run(run_path)


@cli.command('update_analysis_run')
@click.argument('analysis_run_id')
@click.option('--downstream_name', default=None)
@click.option('--pretrain_name', default=None)
@click.option('--monitor_metric', default=None)
@click.option('--update_summary', default=True)
@click.option('--update_pretrain_summary', default=True)
@click.option('--include_key', multiple=True)
def update_analysis_cmd(analysis_run_id, downstream_name, pretrain_name=None, monitor_metric=None,
                        update_summary=True, update_pretrain_summary=True,
                        include_key=None):
    analysis_run = load_wandb_run_from_id(analysis_run_id)
    if update_summary:
        if monitor_metric is None:
            monitor_metric = analysis_run.config['monitor_metric']
        update_best_epoch(analysis_run, monitor_metric, analysis_run.config['monitor_metric_mode'])

    if pretrain_name is not None:
        pretrain_run = PretrainingRun.from_run_path(pretrain_name)
        pretrain_run.link_with_analysis_run(analysis_run)
    else:
        pretrain_run = PretrainingRun.from_analysis_run(analysis_run)
    if update_pretrain_summary:
        assert downstream_name is not None
        pretrain_run.update_summary(analysis_run.summary, analysis_name=f'downstream/{downstream_name}', analysis_run=analysis_run, include_keys=include_key)


@cli.command('save_model_data')
@click.argument('pretrain_name')
@click.option('--gpu', multiple=True, type=int)
@click.option('--data', default='test')
def save_model_data_cmd(pretrain_name: str, gpu, data='test'):
    run = PretrainingRun.from_run_path(pretrain_name)
    run.compute_model_data(gpus=gpu, data=data)


@cli.command('save_downstream_data')
@click.argument('pretrain_name')
@click.option('--gpu', type=int, default=None)
@click.option('--task', default='rsna_pneunomia_detection')
@click.option('--dataset', default='rsna_seg')
@click.option('--data', default='test')
def save_downstream_data_cmd(pretrain_name: str, gpu, task, dataset, data='test'):
    run = PretrainingRun.from_run_path(pretrain_name)
    run.compute_downstream_data(gpu=gpu, task=task, dataset=dataset, data=data)


@cli.command('save_baseline_data')
@click.argument('run_dir')
@click.option('--config')
@click.option('--gpu', type=int, default=None)
@click.option('--dataset', default='mimic-cxr_all_find-impr')
@click.option('--data', default='test')
def save_baseline_data_cmd(config, run_dir, gpu, dataset, data):
    init_config_store()
    initialize(config_path="../../configs/")
    model_config = compose(config_name=f'experiment/{config}')['pretrain_model']

    compute_untrained_model_data(model_config, run_dir, gpus=[gpu], dataset=dataset, data=data)


@cli.command('save_baseline_downstream_data')
@click.argument('run_dir')
@click.option('--encoder_config_name', default='resnet50_imagenet')
@click.option('--gpu', type=int, default=None)
@click.option('--task', default='rsna_pneunomia_detection')
@click.option('--dataset', default='rsna_seg')
@click.option('--data', default='test')
def save_baseline_downstream_data_cmd(encoder_config_name, run_dir, task, dataset,
                    data='test', gpu=None):
    cs = ConfigStore.instance()
    cs.store(group="scan_encoder", name="base_scan_encoder", node=ScanEncoderConfig)
    initialize(config_path="../../configs/")
    encoder_config = compose(config_name=f'scan_encoder/{encoder_config_name}')['scan_encoder']

    predictions_dir = os.path.join(run_dir, 'predictions', f'downstream_{dataset}_{data}')
    LocalScanDownstreamEmbeddingsExporter.export_baseline(encoder_config,
                                                          predictions_dir=predictions_dir,
                                                          segmentation_task=task,
                                                          dataset=dataset, data=data, gpu=gpu)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    cli()

