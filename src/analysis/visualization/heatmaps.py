import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)

from analysis.postprocess_run import PretrainingRun


def plot_scan_region_weights_heatmap(run: PretrainingRun, data='test'):
    all_weights = []
    size = None

    log.info('Collecting weights...')
    mean_image = None
    num_batches = 0
    for batch in tqdm(run.iter_model_data_batches(data, load_inputs=True)):
        batch_mean_image = batch.inputs.x_a.scan.float().mean(dim=0)  # H x W
        if mean_image is None:
            mean_image = batch_mean_image
        else:
            mean_image = mean_image + batch_mean_image
        num_batches += 1

        all_weights.append(batch.embeddings.weights_a)  # (B x N)
        new_size = batch.embeddings.local_size_a
        assert size is None or size == new_size, f'{size} != {new_size}'
        size = new_size
    mean_image = mean_image / num_batches
    mean_image = mean_image - mean_image.min()
    mean_image = mean_image / mean_image.max()

    avg_weights = torch.cat(all_weights, dim=0).mean(0)  # (N)
    avg_weights = avg_weights.view(*size)

    log.info('Plotting...')
    fig, ax = plt.subplots()
    ax = sns.heatmap(avg_weights, linewidth=0.5,
                alpha=0.3,
                zorder=2,
                ax=ax)
    ax.imshow(mean_image,
                cmap='gray',
                aspect=ax.get_aspect(),
                extent=ax.get_xlim() + ax.get_ylim(),
                zorder=1)
    plt.tight_layout()
    plot_folder = os.path.join(run.run_path, 'plots')
    os.makedirs(plot_folder, exist_ok=True)
    path = os.path.join(plot_folder, f'scan_weights_heatmap_{data}.pdf')
    plt.savefig(path)
    log.info(f'Plot saved to {path}')
