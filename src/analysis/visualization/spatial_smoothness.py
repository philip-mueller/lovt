import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import tqdm
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)
from analysis.postprocess_run import PretrainingRun


def plot_spatial_smoothness(run: PretrainingRun, dataset=None, data='test', plot_zl=False, limit_samples=None):
    log.info('Computing distances for spatial smoothness...')
    cos_emb = []
    coord_dists = []
    batch_iterator = run.iter_model_data_batches(data, load_attentions=False) if dataset is None \
        else run.iter_downstream_data_batches(dataset, data)
    num_samples = 0
    for batch in tqdm(batch_iterator):
        if dataset is None:
            if plot_zl:
                emb = batch.embeddings.zl_a
            else:
                emb = batch.embeddings.yl_a  # (B x N_a x d)
            H, W = batch.inputs.x_a.local_regions_shape
        else:
            emb = batch.yl
            H, W = batch.x.local_regions_shape

        num_samples += emb.shape[0]
        if limit_samples is not None and num_samples > limit_samples:
            emb = emb[:limit_samples - num_samples]

        B, N_a, _ = emb.size()
        emb = F.normalize(emb, dim=-1)
        coordinates = torch.meshgrid(torch.arange(H), torch.arange(W))
        coordinates = torch.stack(coordinates).float()  # (2 x H x W)
        coordinates = coordinates.view(2, -1).T  # ((H*W) x 2)
        coordinates /= math.sqrt(H ** 2 + W ** 2)
        distances = torch.cdist(coordinates, coordinates)  # (N_a x N_a)
        assert distances.size() == (N_a, N_a)
        coord_dists.append(distances[None, :, :].expand(B, -1, -1))
        cos_emb.append(torch.bmm(emb, emb.transpose(-1, -2)))  # (B x N_a x N_a)

        if limit_samples is not None and num_samples >= limit_samples:
            break
    coord_dists = torch.cat(coord_dists, dim=0).flatten().numpy()
    cos_emb = torch.cat(cos_emb, dim=0).flatten().numpy()

    log.info('Computing means...')
    df = pd.DataFrame({'coord_dists': coord_dists, 'cos_emb': cos_emb})
    df['coord_dists'] = df['coord_dists'].round(2)
    df_by_ccoord_dists: pd.DataFrame = df.groupby('coord_dists')\
        .mean()\
        .sort_index()
    mean_cos_a = df_by_ccoord_dists['cos_emb'].values
    ordered_coord_dists = df_by_ccoord_dists.index.values

    log.info('Plotting...')
    fig, ax = plt.subplots()
    ax.scatter(x=coord_dists, y=cos_emb, alpha=0.5, marker='.')
    ax.plot(ordered_coord_dists, mean_cos_a, c='tab:orange', alpha=0.3, linestyle='--')

    ax.set_xlabel('Spatial Distance of Region Pair')
    ax.set_ylabel('Cosine Similarity of Region Embeddings Pair')
    plt.tight_layout()
    plot_folder = os.path.join(run.run_path, 'plots')
    os.makedirs(plot_folder, exist_ok=True)
    emb_name = 'zl' if plot_zl else 'yl'
    dataset_infix = dataset + '_' if dataset is not None else ''
    path = os.path.join(plot_folder, f'spatial_smoothness_{emb_name}_{dataset_infix}{data}.png')
    plt.savefig(path, dpi=1000)
    #plt.show()
    log.info(f'Plot saved to {path}')
