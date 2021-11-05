import csv
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Iterable

import numpy as np
from datasets import tqdm
import pandas as pd

if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.append(str(root_path))

log = logging.getLogger(__name__)

from analysis.postprocess_run import load_wandb_run_from_id

BASE_PATH = Path(os.path.realpath(__file__)).absolute().parent.parent.parent
RESULTS_PATH = os.path.join(BASE_PATH, 'results')


def get_paper_data(category=None):
    data = pd.read_csv(os.path.join(RESULTS_PATH, 'runs_paper.csv'))
    data = data.set_index('name')

    if category is not None:
        if isinstance(category, str) or not isinstance(category, Iterable):
            category = [category]
        else:
            category = [str(cat) for cat in category]
        if len(category) > 0:
            def filter_category(x):
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    return True
                x = str(x)
                if len(x) == 0:
                    return True
                row_categories = x.split(';')
                return any(cat in row_categories for cat in category)

            data = data.loc[data['category'].apply(filter_category)]
    return data


def get_run_definitions_dict() -> List[dict]:
    path = os.path.join(RESULTS_PATH, 'runs.csv')
    if not os.path.exists(path):
        return []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def get_run_definitions_df() -> pd.DataFrame:
    path = os.path.join(RESULTS_PATH, 'runs.csv')
    if not os.path.exists(path):
        data = pd.DataFrame(columns=['name', 'run_id', 'run_folder'])
    else:
        data: pd.DataFrame = pd.read_csv(path, dtype=str)
    data = data.set_index('name')
    return data


def update_run_definitions(run_name, run_id=None, **kwargs):
    data = get_run_definitions_df()
    if run_name not in data.index:
        data.at[run_name, 'run_folder'] = run_name
    if run_id is not None:
        data.at[run_name, 'run_id'] = run_id
    for k, v in kwargs.items():
        data.at[run_name, k] = v
    data.to_csv(os.path.join(RESULTS_PATH, 'runs.csv'))


def load_results(results_name: str) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(os.path.join(RESULTS_PATH, 'generated', f'{results_name}.csv'))
    data = data.set_index('name')
    return data


def update_results(results_name, run_name, **kwargs):
    data = load_results(results_name)
    for k, v in kwargs.items():
        data.at[run_name, k] = v
    data.to_csv(os.path.join(RESULTS_PATH, 'generated', f'{results_name}.csv'))


def load_analysis_results() -> pd.DataFrame:
    data = pd.read_csv(os.path.join(RESULTS_PATH, 'generated', f'analysis-test.csv'))
    data = data.set_index('name')
    return data


def export_analysis_results(recompute_run=None, reexport_all=False):
    results = load_analysis_results()
    runs = get_runs_to_reexport(results, recompute_run, reexport_all)

    runs = runs[~runs.isna()['run_id']]
    log.info(f'Exporting analysis results of the following runs: {runs.index}')
    for name, run_def in tqdm(runs.iterrows()):
        run = load_wandb_run_from_id(run_def['run_id'])

        run_results = {key[len('analysis-test/'):]: value
                       for key, value in run.summary.items()
                       if key.startswith('analysis-test/')}
        if name not in results.index:
            results.loc[name] = pd.Series(run_results)
        else:
            results.loc[name].update(pd.Series(run_results))

    if len(runs.index) > 0:
        results.to_csv(os.path.join(RESULTS_PATH, 'generated', f'analysis-test.csv'))


def load_downstream_results(downstream_task) -> pd.DataFrame:
    path = os.path.join(RESULTS_PATH, 'generated', f'downstream_{downstream_task}.csv')
    if os.path.exists(path):
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame(columns=['name'])
    data = data.set_index('name')
    return data


DOWNSTREAM_METRICS = {
    'rsna_frozen_1': ['rsna_1_test/mAP', 'rsna_1_test/mAP@0.5', 'rsna_1_test/mAP@0.75'],
    'rsna_finetune_1': ['rsna_finetune_1_test/mAP', 'rsna_finetune_1_test/mAP@0.5', 'rsna_finetune_1_test/mAP@0.75'],
    'rsna_seg_linear_1': ['rsna_seg_lin_1_test/opacity_dice'],
    'rsna_frozen_10': ['rsna_test/mAP', 'rsna_test/mAP@0.5', 'rsna_test/mAP@0.75'],
    'rsna_finetune_10': ['rsna_finetune_test/mAP', 'rsna_finetune_test/mAP@0.5', 'rsna_finetune_test/mAP@0.75'],
    'rsna_seg_linear_10': ['rsna_seg_lin_test/opacity_dice'],
    'rsna_frozen_100': ['rsna_full_test/mAP', 'rsna_full_test/mAP@0.5', 'rsna_full_test/mAP@0.75'],
    'rsna_finetune_100': ['rsna_finetune_full_test/mAP', 'rsna_finetune_full_test/mAP@0.5', 'rsna_finetune_full_test/mAP@0.75'],
    'rsna_seg_linear_100': ['rsna_seg_lin_full_test/opacity_dice'],
    'covid_unet_frozen': ['covid_unet_frozen_test/opacity_dice'],
    'covid_unet_finetune': ['covid_unet_finetune_test/opacity_dice'],
    'covid_linear': ['covid_lin_test/opacity_dice'],
    'nih_seg_linear': ['nih_seg_lin_test/avg_dice',
                       'nih_seg_lin_test/Atelectasis_dice', 'nih_seg_lin_test/Cardiomegaly_dice',
                       'nih_seg_lin_test/Effusion_dice', 'nih_seg_lin_test/Infiltrate_dice',
                       'nih_seg_lin_test/Mass_dice', 'nih_seg_lin_test/Nodule_dice',
                       'nih_seg_lin_test/Pneumonia_dice', 'nih_seg_lin_test/Pneumothorax_dice'],
    'pneumothorax_unet_frozen_100': ['pneumo_unet_frozen_full_test/pneumothorax_dice'],
    'pneumothorax_unet_finetune_100': ['pneumo_unet_finetune_full_test/pneumothorax_dice'],
    'object_frozen_100': ['object_frozen_full_test/froc', 'object_frozen_full_test/mAP', 'object_frozen_full_test/mAP@0.5'],
    'object_finetune_100': ['object_finetune_full_test/froc', 'object_finetune_full_test/mAP', 'object_finetune_full_test/mAP@0.5'],
    'object_seg_linear_100': ['object_seg_lin_full_test/avg_dice']
}


def export_downstream_results(downstream_task, recompute_run=None, reexport_all=False):
    results = load_downstream_results(downstream_task)
    runs = get_runs_to_reexport(results, recompute_run, reexport_all)

    metrics = DOWNSTREAM_METRICS[downstream_task]

    log.info(f'Exporting the downstream results of the following runs: {runs.index}')
    for name, run_def in tqdm(runs.iterrows()):
        run_results = export_multi_run_results(run_def[downstream_task], metrics)
        new_results = pd.DataFrame(index=[name], data=run_results)
        new_results.index.name = 'name'
        results = new_results.combine_first(results)

    if len(runs.index) > 0:
        results.to_csv(os.path.join(RESULTS_PATH, 'generated', f'downstream_{downstream_task}.csv'))


def get_runs_to_reexport(results, recompute_run, reexport_all):
    runs = get_run_definitions_df()
    if not reexport_all:
        new_runs_mask = ~runs.index.isin(results.index)
        if recompute_run is not None:
            new_runs_mask = new_runs_mask | (runs.index == recompute_run)
        runs = runs[new_runs_mask]
    return runs


MULTIRUN_RESULT_SUB_METRICS = ['run_results', 'mean', 'std', 'num_runs', '95_interval']


def export_multi_run_results(run_ids: str, metrics: list):
    if run_ids is None or (isinstance(run_ids, float) and math.isnan(run_ids)):
        return {}
    run_ids = run_ids.split(';')
    metric_results = defaultdict(list)
    for run_id in run_ids:
        run = load_wandb_run_from_id(run_id)
        for metric in metrics:
            metric_results[metric].append(run.summary[metric])
    results = {}
    for metric in metrics:
        values = np.array(metric_results[metric], dtype=float)
        results[f'{metric}_run_results'] = ';'.join(str(value) for value in values)
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values, ddof=1)
        results[f'{metric}_num_runs'] = len(values)
        results[f'{metric}_95_interval'] = 1.96 * (results[f'{metric}_std'] / math.sqrt(results[f'{metric}_num_runs']))
    return results
