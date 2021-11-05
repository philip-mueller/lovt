import logging
import os
import sys
from pathlib import Path
from pprint import pformat
from typing import Union

import click
from tqdm import tqdm

from analysis.data_exporter import update_results

if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.append(str(root_path))

log = logging.getLogger(__name__)

from analysis.postprocess_run import PretrainingRun, BaselineRun
from metrics.embedding_metrics import EmbeddingMetrics, DownstreamEmbeddingMetrics


def run_embedding_analysis(run: PretrainingRun, data='test'):
    log.info('Running (basic) embedding analysis...')
    embedding_metrics = EmbeddingMetrics(compute_cov=True)
    for batch in tqdm(run.iter_model_data_batches(data=data, load_inputs=False, load_attentions=False)):
        embedding_metrics(batch.embeddings)
    log.info('Computing metrics...')
    results = embedding_metrics.compute()
    log.info(f'Results: {pformat(results)}')
    log.info('Saving results to run...')
    run.update_summary(results, analysis_name=f'analysis-{data}/emb')
    log.info('Done')


def run_downstream_embedding_analysis(run: Union[BaselineRun, PretrainingRun], dataset, data='test'):
    log.info('Running baseline embedding analysis...')
    embedding_metrics = DownstreamEmbeddingMetrics(compute_cov=True)
    for batch in tqdm(run.iter_downstream_data_batches(dataset=dataset, data=data)):
        embedding_metrics(batch)
    log.info('Computing metrics...')
    results = embedding_metrics.compute()
    log.info(f'Results: {pformat(results)}')
    log.info('Saving results...')

    update_results(f'analysis-{dataset}', run.baseline_name, **results)
    log.info('Done')
