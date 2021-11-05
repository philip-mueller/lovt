import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra


if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.insert(0, str(root_path))


from models.image.scan_encoder import ScanEncoderConfig
from models.text.report_encoder import ReportEncoderConfig
from scripts.run_finetuning_baseline import EncoderFinetuningExperimentConfig
from analysis.visualization.heatmaps import plot_scan_region_weights_heatmap
from analysis.data_exporter import export_downstream_results, update_run_definitions, \
    export_analysis_results, update_results
from analysis.embedding_analysis import run_embedding_analysis, run_downstream_embedding_analysis
from analysis.postprocess_run import PretrainingRun, BaselineRun
from analysis.visualization.downstream_plotter import plot_downstream_results
from analysis.visualization.emb_properties_vis import plot_emb_properties
from analysis.visualization.embedding_vis import plot_embeddings, plot_downstream_embeddings
from analysis.visualization.spatial_smoothness import plot_spatial_smoothness
from models.downstream.downstream_evaluation import ClassificationEvaluationModelConfig, \
    SegmentationEvaluationModelConfig, DetectionEvaluationModelConfig
from scripts.run_finetuning import CheckpointFinetuningExperimentConfig

log = logging.getLogger(__name__)

EXPERIMENT_CONFIGS = {
    'rsna_frozen_10': 'eval/eval_rsna_frozen',
    'rsna_finetune_10': 'eval/eval_rsna_finetune',
    'rsna_seg_linear_10': 'eval/eval_rsna_seg_frozen',
    'rsna_frozen_100': 'eval/eval_rsna_frozen_full',
    'rsna_finetune_100': 'eval/eval_rsna_finetune_full',
    'rsna_seg_linear_100': 'eval/eval_rsna_seg_frozen_full',
    'rsna_frozen_1': 'eval/eval_rsna_frozen_1',
    'rsna_finetune_1': 'eval/eval_rsna_finetune_1',
    'rsna_seg_linear_1': 'eval/eval_rsna_seg_frozen_1',
    'covid_unet_frozen': 'eval/eval_covid_unet_frozen',
    'covid_unet_finetune': 'eval/eval_covid_unet_finetune',
    'covid_linear': 'eval/eval_covid_lin_frozen',
    'nih_seg_linear': 'eval/eval_nih_seg_frozen',
    'pneumothorax_unet_frozen_100': 'eval/eval_pneumo_unet_frozen_full',
    'pneumothorax_unet_finetune_100': 'eval/eval_pneumo_unet_finetune_full',
    'object_frozen_100': 'eval/eval_object_frozen_full',
    'object_finetune_100': 'eval/eval_object_finetune_full',
    'object_seg_linear_100': 'eval/eval_object_seg_frozen_full',
}

BASELINE_EXPERIMENT_CONFIGS = {
    'rsna_frozen_10': 'eval/baseline_random_rsna_frozen',
    'rsna_finetune_10': 'eval/baseline_random_rsna_finetune',
    'rsna_seg_linear_10': 'eval/baseline_random_rsna_seg_frozen',
    'rsna_frozen_100': 'eval/baseline_random_rsna_frozen_full',
    'rsna_finetune_100': 'eval/baseline_random_rsna_finetune_full',
    'rsna_seg_linear_100': 'eval/baseline_random_rsna_seg_frozen_full',
    'rsna_frozen_1': 'eval/baseline_random_rsna_frozen_1',
    'rsna_finetune_1': 'eval/baseline_random_rsna_finetune_1',
    'rsna_seg_linear_1': 'eval/baseline_random_rsna_seg_frozen_1',
    'covid_unet_frozen': 'eval/baseline_random_covid_unet_frozen',
    'covid_unet_finetune': 'eval/baseline_random_covid_unet_finetune',
    'covid_linear': 'eval/baseline_random_covid_lin_frozen',
    'nih_seg_linear': 'eval/baseline_random_nih_seg_frozen',
    'pneumothorax_unet_frozen_100': 'eval/baseline_random_pneumo_unet_frozen_full',
    'pneumothorax_unet_finetune_100': 'eval/baseline_random_pneumo_unet_finetune_full',
    'object_frozen_100': 'eval/baseline_random_object_frozen_full',
    'object_finetune_100': 'eval/baseline_random_object_finetune_full',
    'object_seg_linear_100': 'eval/baseline_random_object_seg_frozen_full',
}

EVAL_EXPERIMENTS = [
    # name, average_runs, learning_rates, tuned metric
    ('rsna_frozen_10', 5, None, None),
]

EXTENDED_EVAL_EXPERIMENTS = [
    ('rsna_finetune_10', 5, [3e-4, 1e-3], 'rsna_finetune_val/mAP'),
    ('rsna_seg_linear_10', 5, [3e-2, 1e-2, 3e-3, 1e-3], 'rsna_seg_lin_val/opacity_dice'),

    ('rsna_frozen_100', 5, [3e-4, 1e-3], 'rsna_full_val/mAP'),
    ('rsna_finetune_100', 5, [3e-4, 1e-4], 'rsna_finetune_full_val/mAP'),
    ('rsna_seg_linear_100', 5, [3e-2, 1e-2, 3e-3, 1e-3], 'rsna_seg_lin_full_val/opacity_dice'),

    ('rsna_frozen_1', 5, [1e-3, 3e-4], 'rsna_1_val/mAP'),
    ('rsna_finetune_1', 5, [3e-4, 1e-4], 'rsna_finetune_1_val/mAP'),
    ('rsna_seg_linear_1', 5, [3e-2, 1e-2, 3e-3, 1e-4], 'rsna_seg_lin_1_val/opacity_dice'),

    ('covid_unet_frozen', 5, [1e-4, 3e-4, 1e-3], 'covid_unet_frozen_val/opacity_dice'),
    ('covid_unet_finetune', 5, [1e-4, 3e-4], 'covid_unet_finetune_val/opacity_dice'),
    ('covid_linear', 5, [3e-2, 1e-2, 3e-3, 1e-3], 'covid_lin_val/opacity_dice'),

    ('nih_seg_linear', 5, [1e-1, 3e-2, 1e-2], 'nih_seg_lin_val/avg_dice'),

    ('pneumothorax_unet_frozen_100', 5, [3e-5, 1e-5], 'pneumo_unet_frozen_full_val/pneumothorax_dice'),
    ('pneumothorax_unet_finetune_100', 5, [3e-5, 1e-5], 'pneumo_unet_finetune_full_val/pneumothorax_dice'),

    ('object_frozen_100', 5, [1e-3, 3e-3], 'object_frozen_full_val/froc'),
    ('object_finetune_100', 5, [1e-3, 3e-4], 'object_finetune_full_val/froc'),
    ('object_seg_linear_100', 5, [3e-1, 1e-1, 3e-2, 1e-2], 'object_seg_lin_full_val/avg_dice'),
]


def run_postprocessing_and_evaluation(pretrained_model_name,
                                      gpus,
                                      category=None,
                                      evaluate_basic=False, evaluate_extended=False,
                                      analyze_embeddings=False,
                                      export_data=False, export_downstream=False,
                                      create_data_plots=False, create_downstream_plots=False, recreate_shared_plots=False):
    run = PretrainingRun.from_run_path(pretrained_model_name)
    assert run.run.name == pretrained_model_name, f'{run.run.name} was not {pretrained_model_name}'

    update_run_definitions(pretrained_model_name, run_id=run.run.id)

    # ----- Running evaluations -----
    src_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    sys.path.append(str(src_path))
    sys.path.append(str(os.path.join(src_path, 'scripts')))
    if evaluate_basic:
        run_finetuning_experiments(pretrained_model_name, EVAL_EXPERIMENTS)
    if evaluate_extended:
        run_finetuning_experiments(pretrained_model_name, EXTENDED_EVAL_EXPERIMENTS)

    # ----- Computing metrics, exporting data, creating plots -----
    if export_data:
        run.compute_model_data(gpus=[gpus], data='test')
    if export_downstream:
        run.compute_downstream_data(gpu=gpus, task='rsna_pneunomia_detection', dataset='rsna_seg', data='test')
    if analyze_embeddings:
        run_embedding_analysis(run, data='test')
    if analyze_embeddings or recreate_shared_plots:
        export_analysis_results(recompute_run=pretrained_model_name)

    if create_data_plots:
        plot_spatial_smoothness(run, limit_samples=100)
        plot_embeddings(run)
        plot_scan_region_weights_heatmap(run)
    if create_downstream_plots:
        plot_downstream_embeddings(run)
        plot_spatial_smoothness(run, dataset='rsna_seg', limit_samples=100)
    if recreate_shared_plots:
        plot_all_downstream_results(category)
        plot_emb_properties(category=category)


def run_baseline_postprocessing_and_evaluation(baseline_model_name,
                                               gpus,
                                               pretrain_dataset=None,
                                               evaluate_basic=False, evaluate_extended=False,
                                               analyze_embeddings=False,
                                               export_data=False, export_downstream=False,
                                               create_data_plots=False, create_downstream_plots=False):
    run = BaselineRun(baseline_model_name)

    if run.run is not None:
        assert run.run.name == baseline_model_name, f'{run.run.name} was not {baseline_model_name}'
        update_run_definitions(baseline_model_name, run_id=run.run.id)

    # ----- Running evaluations -----
    src_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    sys.path.append(str(src_path))
    sys.path.append(str(os.path.join(src_path, 'scripts')))
    if evaluate_basic:
        run_finetuning_experiments(baseline_model_name, EVAL_EXPERIMENTS, baseline=True)
    if evaluate_extended:
        run_finetuning_experiments(baseline_model_name, EXTENDED_EVAL_EXPERIMENTS, baseline=True)

    # ----- Computing metrics, exporting data, creating plots -----
    if export_data:
        run.compute_downstream_data(gpu=gpus, task=None, dataset=pretrain_dataset, data='test')
    if export_downstream:
        run.compute_downstream_data(gpu=gpus, task='rsna_pneunomia_detection', dataset='rsna_seg', data='test')
    if analyze_embeddings:
        run_downstream_embedding_analysis(run, dataset=pretrain_dataset, data='test')

    if create_data_plots:
        plot_spatial_smoothness(run, dataset=pretrain_dataset, limit_samples=100)
    if create_downstream_plots:
        plot_downstream_embeddings(run)
        plot_spatial_smoothness(run, dataset='rsna_seg', limit_samples=100)


def plot_shared(category=None, skip_alignment=False):
    plot_all_downstream_results(category)
    plot_emb_properties(category=category, skip_alignment=skip_alignment)


def plot_all_downstream_results(category):
    plot_downstream_results('rsna_frozen_10', metric='rsna_test/mAP', ylabel='mAP (%)', category=category)
    plot_downstream_results('rsna_finetune_10', metric='rsna_finetune_test/mAP', ylabel='mAP (%)', category=category)
    plot_downstream_results('rsna_seg_linear_10', metric='rsna_seg_lin_test/opacity_dice', ylabel='Dice (%)', category=category)

    plot_downstream_results('rsna_frozen_100', metric='rsna_full_test/mAP', ylabel='mAP (%)', category=category)
    plot_downstream_results('rsna_finetune_100', metric='rsna_finetune_full_test/mAP', ylabel='mAP (%)', category=category)
    plot_downstream_results('rsna_seg_linear_100', metric='rsna_seg_lin_full_test/opacity_dice', ylabel='Dice (%)', category=category)

    plot_downstream_results('rsna_frozen_1', metric='rsna_1_test/mAP', ylabel='mAP (%)', category=category)
    plot_downstream_results('rsna_finetune_1', metric='rsna_finetune_1_test/mAP', ylabel='mAP (%)', category=category)
    plot_downstream_results('rsna_seg_linear_1', metric='rsna_seg_lin_1_test/opacity_dice', ylabel='Dice (%)', category=category)

    plot_downstream_results('covid_unet_frozen', metric='covid_unet_frozen_test/opacity_dice', ylabel='Dice (%)', category=category)
    plot_downstream_results('covid_unet_finetune', metric='covid_unet_finetune_test/opacity_dice', ylabel='Dice (%)', category=category)
    plot_downstream_results('covid_linear', metric='covid_lin_test/opacity_dice', ylabel='Dice (%)',
                            category=category)

    plot_downstream_results('nih_seg_linear', metric='nih_seg_lin_test/avg_dice', ylabel='Avg Dice (%)',
                            category=category)

    plot_downstream_results('pneumothorax_unet_frozen_100', metric='pneumo_unet_frozen_full_test/pneumothorax_dice', ylabel='Dice (%)',
                            category=category)
    plot_downstream_results('pneumothorax_unet_finetune_100', metric='pneumo_unet_finetune_full_test/pneumothorax_dice', ylabel='Dice (%)',
                            category=category)

    plot_downstream_results('object_frozen_100', metric='object_frozen_full_test/froc', ylabel='fROC (%)', category=category)
    plot_downstream_results('object_finetune_100', metric='object_finetune_full_test/froc', ylabel='fROC (%)',
                            category=category)
    plot_downstream_results('object_seg_linear_100', metric='object_seg_lin_full_test/avg_dice', ylabel='Dice (%)',
                            category=category)


def run_finetuning_experiments(pretrained_model_name, experiments, baseline=False):
    for downstream_name, average_runs, learning_rates, metric in experiments:
        experiment_config_name = BASELINE_EXPERIMENT_CONFIGS[downstream_name] if baseline \
            else EXPERIMENT_CONFIGS[downstream_name]
        log.info(f'Running downstream experiment {downstream_name} using config {experiment_config_name}')
        run_finetuning_experiment(pretrained_model_name, downstream_name, experiment_config_name,
                                  average_runs, learning_rates, metric,
                                  baseline=baseline)


def run_finetuning_experiment(pretrained_model_name, name, experiment_config_name, average_runs, lrs=None, metric=None,
                              baseline=False):
    if baseline:
        from scripts.run_finetuning_baseline import run_finetuning
    else:
        from scripts.run_finetuning import run_finetuning
    cs = ConfigStore.instance()
    cs.store(name="fine_tune_config", node=CheckpointFinetuningExperimentConfig)
    cs.store(group="evaluation_model", name="classification_eval", node=ClassificationEvaluationModelConfig)
    cs.store(group="evaluation_model", name="segmentation_eval", node=SegmentationEvaluationModelConfig)
    cs.store(group="evaluation_model", name="detection_eval", node=DetectionEvaluationModelConfig)
    cs.store(name="baseline_fine_tune_config", node=EncoderFinetuningExperimentConfig)
    cs.store(group="encoder", name="base_scan_encoder", node=ScanEncoderConfig)
    cs.store(group="encoder", name="base_report_encoder", node=ReportEncoderConfig)
    GlobalHydra.instance().clear()
    initialize(config_path="../../configs/")
    if baseline:
        config: EncoderFinetuningExperimentConfig = compose(
            config_name='baseline_fine_tune',
            overrides=[f'+experiment={experiment_config_name}'])
        if pretrained_model_name == 'ImageNet':
            config.encoder.backbone_pretrained = True
        elif pretrained_model_name == 'random':
            config.encoder.backbone_pretrained = False
        else:
            config.encoder.backbone_pretrained = False
            config.encoder.backbone_checkpoint = f'baseline:{pretrained_model_name}'
    else:
        config: CheckpointFinetuningExperimentConfig = compose(config_name='fine_tune',
                                                               overrides=[f'+experiment={experiment_config_name}'])
        config.pretrained_model = pretrained_model_name

    config.name = name + '_' + pretrained_model_name
    if lrs is not None and len(lrs) > 1:
        # find best lr with average_runs = 1

        metric = f'{metric}__runs_mean'

        def is_run_better(new_summary, old_summary):
            assert metric in new_summary and metric in old_summary, \
                f'{metric}, {list(old_summary.keys())}, {list(new_summary.keys())}'
            return new_summary[metric] > old_summary[metric]

        summary = None
        best_lr = None
        for lr in lrs:
            create_work_dir(name, pretrained_model_name)
            log.info(f'Running finetuning with lr {lr}')
            config.evaluation_model.learning_rate = lr
            config.average_runs = 1
            current_summary = run_finetuning(config)
            assert current_summary is not None
            if summary is None or is_run_better(current_summary, summary):
                summary = current_summary
                best_lr = lr
        config.evaluation_model.learning_rate = best_lr
    if lrs is not None and len(lrs) == 1:
        config.evaluation_model.learning_rate = lrs[0]
    if lrs is None or average_runs > 1:
        # now run with best lr and the specified average_runs
        create_work_dir(name, pretrained_model_name)
        config.average_runs = average_runs
        summary = run_finetuning(config)

    run_ids = summary['run_ids']
    update_run_definitions(pretrained_model_name, **{name: run_ids})
    export_downstream_results(name, recompute_run=pretrained_model_name)


def create_work_dir(name, pretrained_model_name):
    path = f'logs/downstream_runs/{pretrained_model_name}/{name}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(path)
    os.chdir(path)


@click.group()
def cli():
    pass


@cli.command('evaluate_all')
@click.argument('pretrained_model_name')
@click.option('--gpu', type=int, default=None)
@click.option('--category', multiple=True, type=str)
@click.option('--skip_downstream/--no-skip_downstream', default=False)
@click.option('--evaluate_extended/--no-evaluate_extended', default=True)
@click.option('--export_downstream/--no-export_downstream', default=False)
def evaluate_all(pretrained_model_name, gpu, category, evaluate_extended=False,
                 export_downstream=False, skip_downstream=False):
    run_postprocessing_and_evaluation(pretrained_model_name,
                                      gpu,
                                      category=category,
                                      evaluate_basic=not skip_downstream,
                                      evaluate_extended=not skip_downstream and evaluate_extended,
                                      analyze_embeddings=True,
                                      export_data=True, export_downstream=export_downstream,
                                      create_data_plots=True, create_downstream_plots=export_downstream,
                                      recreate_shared_plots=True)


@cli.command('evaluate_downstream')
@click.argument('pretrained_model_name')
@click.option('--evaluate_basic/--no-evaluate_basic', default=True)
@click.option('--evaluate_extended/--no-evaluate_extended', default=False)
def evaluate_downstream(pretrained_model_name, evaluate_basic=True, evaluate_extended=False):
    run_postprocessing_and_evaluation(pretrained_model_name,
                                      gpus=None,
                                      category=None,
                                      evaluate_basic=evaluate_basic, evaluate_extended=evaluate_extended,
                                      analyze_embeddings=False,
                                      export_data=False, export_downstream=False,
                                      create_data_plots=False, create_downstream_plots=False,
                                      recreate_shared_plots=False)


@cli.command('evaluate_baseline_downstream')
@click.argument('baseline_model_name')
@click.option('--evaluate_basic/--no-evaluate_basic', default=True)
@click.option('--evaluate_extended/--no-evaluate_extended', default=False)
def evaluate_baseline_downstream(baseline_model_name, evaluate_basic=True, evaluate_extended=False):
    run_baseline_postprocessing_and_evaluation(baseline_model_name,
                                      gpus=None,
                                      evaluate_basic=evaluate_basic, evaluate_extended=evaluate_extended,
                                      analyze_embeddings=False,
                                      export_data=False, export_downstream=False,
                                      create_data_plots=False, create_downstream_plots=False)


@cli.command('analyze')
@click.argument('pretrained_model_name')
@click.option('--gpu', type=int, default=None)
@click.option('--category', multiple=True, type=str)
@click.option('--export/--no-export', default=False)
def analyze(pretrained_model_name, gpu, category, export=False):
    run_postprocessing_and_evaluation(pretrained_model_name,
                                      gpu,
                                      category=category,
                                      analyze_embeddings=True,
                                      export_data=export, export_downstream=False,
                                      create_data_plots=False, create_downstream_plots=False,
                                      recreate_shared_plots=False)


@cli.command('analyze_baseline')
@click.argument('baseline_model_name')
@click.option('--gpu', type=int, default=None)
@click.option('--export/--no-export', default=False)
def analyze_baseline(baseline_model_name, gpu, export=False):
    run_baseline_postprocessing_and_evaluation(baseline_model_name,
                                               gpu,
                                               analyze_embeddings=True,
                                               export_data=export, export_downstream=False,
                                               create_data_plots=False, create_downstream_plots=False)


@cli.command('plot')
@click.argument('pretrained_model_name')
@click.option('--gpu', type=int, default=None)
@click.option('--category', multiple=True, type=str)
@click.option('--export/--no-export', default=False)
@click.option('--data_plots/--no-data_plots', default=False)
@click.option('--downstream_plots/--no-downstream_plots', default=False)
@click.option('--shared_plots/--no-shared_plots', default=False)
def plot(pretrained_model_name, gpu, category, export=False, data_plots=False, downstream_plots=False, shared_plots=False):
    run_postprocessing_and_evaluation(pretrained_model_name,
                                      gpu,
                                      category=category,
                                      analyze_embeddings=False,
                                      export_data=export and data_plots, export_downstream=export and downstream_plots,
                                      create_data_plots=data_plots, create_downstream_plots=downstream_plots,
                                      recreate_shared_plots=shared_plots)


@cli.command('plot_baseline')
@click.argument('baseline_model_name')
@click.option('--gpu', type=int, default=None)
@click.option('--export/--no-export', default=False)
@click.option('--data_plots/--no-data_plots', default=False)
@click.option('--downstream_plots/--no-downstream_plots', default=False)
@click.option('--shared_plots/--no-shared_plots', default=False)
def plot(baseline_model_name, gpu, export=False, data_plots=False, downstream_plots=False):
    run_baseline_postprocessing_and_evaluation(baseline_model_name,
                                               gpu,
                                               analyze_embeddings=False,
                                               export_data=export and data_plots, export_downstream=export and downstream_plots,
                                               create_data_plots=data_plots, create_downstream_plots=downstream_plots,)


@cli.command('plot_shared')
@click.option('--category', multiple=True, type=str)
@click.option('--export/--no-export', default=False)
@click.option('--skip_alignment/--no-skip_alignment', default=False)
def plot(category, export=False, skip_alignment=False):
    if export:
        export_analysis_results()
    plot_shared(category, skip_alignment=skip_alignment)


@cli.command('export')
@click.argument('pretrained_model_name')
def export(pretrained_model_name):
    for downstream_name, experiment_config_name, learning_rates, metric in EVAL_EXPERIMENTS:
        export_downstream_results(downstream_name, recompute_run=pretrained_model_name)
    export_analysis_results(recompute_run=pretrained_model_name)


@cli.command('export_extended')
@click.argument('pretrained_model_name')
def export_extended(pretrained_model_name):
    for downstream_name, experiment_config_name, learning_rates, metric in EXTENDED_EVAL_EXPERIMENTS:
        export_downstream_results(downstream_name, recompute_run=pretrained_model_name)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    cli()
