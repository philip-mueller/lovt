import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from sklearn.manifold import TSNE
from tqdm import tqdm


if __name__ == '__main__':
    scripts_path = Path(os.path.realpath(__file__)).absolute().parent
    root_path = scripts_path.parent
    sys.path.append(str(root_path))


from analysis.downstream_embeddings import LocalScanDownstreamEmbeddings
from data.datasets.chexpert.chexpert_dataset import chexpert_labels
from analysis.postprocess_run import PretrainingRun, ModelData

log = logging.getLogger(__name__)


def project_embeddings(embeddings: torch.Tensor, perplexity=30.0):
    embeddings: np.ndarray = embeddings.cpu().numpy()  # (B x d)
    return TSNE(n_components=2, perplexity=perplexity, init='pca', verbose=1).fit_transform(embeddings)  # (B x 2)


class EmbeddingsTsneProjector:
    def __init__(self, cache_folder: str, embeddings_name: str, overwrite_cache=False, perplexity=30.0):
        self.perplexity = perplexity
        self.embeddings = []
        self.cache_path = os.path.join(cache_folder, f'{embeddings_name}_projected_{perplexity}.pt')
        self.use_cache = os.path.exists(self.cache_path) and not overwrite_cache

    def add_batch(self, embeddings: torch.Tensor, mask=None):
        if self.use_cache:
            return
        if embeddings.ndim == 2:  # global embeddings
            self.embeddings.append(embeddings)
        elif embeddings.ndim == 3:  # local embeddings
            B, N, d = embeddings.size()
            if mask is not None:
                self.embeddings.append(embeddings[mask, :])
            else:
                self.embeddings.append(embeddings.reshape(B*N, d))
        else:
            raise ValueError(embeddings.size())

    def project(self):
        if self.use_cache:
            log.info(f'Using cache for t-SNE embeddings: {self.cache_path}')
            return self.load_cache()
        assert len(self.embeddings) > 0
        embeddings = torch.cat(self.embeddings, dim=0)  # (B_tot*N x d)
        log.info('Projecting embeddings using t-SNE...')
        projected = project_embeddings(embeddings, perplexity=self.perplexity)  # (B_tot*N x 2)
        projected = torch.from_numpy(projected)
        log.info('Projecting done.')
        os.makedirs(Path(self.cache_path).parent, exist_ok=True)
        torch.save(projected, self.cache_path)
        return projected

    def load_cache(self):
        return torch.load(self.cache_path)


class LocalEmbeddingTsnePlotter:
    def __init__(self, model_folder: str, embeddings_name: str, overwrite_cache=False, discrete_colors=False,
                 perplexity=30.):
        self.color_data = []
        self.plots_folder = os.path.join(model_folder, 'plots')
        os.makedirs(self.plots_folder, exist_ok=True)
        self.projector = EmbeddingsTsneProjector(os.path.join(model_folder, 'cached'), embeddings_name, overwrite_cache,
                                                 perplexity=perplexity)
        self.embeddings_name = embeddings_name
        self.discrete_colors = discrete_colors

    def add_batches(self, iterator, fn, limit_samples=None):
        num_samples = 0
        for batch in tqdm(iterator):
            embeddings, color_data, mask = fn(batch)

            if color_data == 'sample_id':
                color_data = torch.arange(num_samples, num_samples + embeddings.shape[0], device=embeddings.device)

            num_samples += embeddings.shape[0]
            if limit_samples is not None and num_samples > limit_samples:
                embeddings = embeddings[:limit_samples - num_samples]
                color_data = color_data[:limit_samples - num_samples] if color_data is not None else None
                mask = mask[:limit_samples - num_samples] if mask is not None else None

            self.add_batch(embeddings, color_data=color_data, mask=mask)

            if limit_samples is not None and num_samples >= limit_samples:
                break

    def add_batch(self, l_embeddings: torch.Tensor, color_data=None, mask=None):
        """

        :param l_embeddings: B x
        :param mask:
        :param color_data:
        :return:
        """
        if color_data is not None:
            if l_embeddings.ndim == 3:  # local
                if color_data.ndim == 1:
                    color_data = color_data.unsqueeze(-1)
                if l_embeddings.shape[:2] != color_data.shape[:2]:
                    color_data = color_data.expand(*l_embeddings.shape[:2])
                if mask is not None:
                    self.color_data.append(color_data[mask])
                else:
                    self.color_data.append(color_data.reshape(-1))
            else:
                assert l_embeddings.ndim == 2
                assert color_data.ndim == 1
                self.color_data.append(color_data)

        self.projector.add_batch(l_embeddings, mask)

    def plot(self, plot_name, colors=None, markers=None, labels=None, colorbar=False):
        projected = self.projector.project()  # (B_tot*N x 2)
        if len(self.color_data) > 0:
            color_data = torch.cat(self.color_data, dim=0).numpy()  # (B_tot*N)
        else:
            color_data = None

        log.info('Plotting data...')
        fig, ax = plt.subplots()
        if self.discrete_colors and color_data is not None:
            for value, (c, marker, label) in enumerate(zip(colors, markers, labels)):
                projected_with_value = projected[color_data == value]
                ax.scatter(projected_with_value[:, 0].numpy(), projected_with_value[:, 1].numpy(), c=c,
                           marker=marker, label=label, alpha=0.3)
                ax.legend(loc='upper left')
        else:
            x = projected[:, 0].numpy()
            y = projected[:, 1].numpy()
            sc = ax.scatter(x, y, c=color_data, alpha=0.3, cmap='viridis')

            if colorbar:
                plt.colorbar(sc)

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.plots_folder, f'tSNE_{plot_name}.pdf')
        plt.savefig(path)
        log.info(f'Saved plot to {path}')
        #plt.show()


def plot_tsne_yl_downstream_class_overlap(run: PretrainingRun, dataset, data='test', label_id=0, limit_samples=None,
                                          perplexity=30., overwrite_cache=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{dataset}_{data}_yl',
                                        overwrite_cache=overwrite_cache, discrete_colors=False, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yl downstream-class-overlap')
    def get_data(batch: LocalScanDownstreamEmbeddings):
        class_probs_yl = batch.class_probs_yl[:, label_id, :, :]  # (B x H x W)
        return batch.yl, class_probs_yl.flatten(1), None

    plotter.add_batches(run.iter_downstream_data_batches(dataset=dataset, data=data), get_data,
                        limit_samples=limit_samples)
    plotter.plot(f'yl_downstream_class_probs_{dataset}', colorbar=True)


def plot_tsne_yl_downstream_sample_class(run: PretrainingRun, dataset, data='test', label_id=0, limit_samples=None,
                                         perplexity=30., overwrite_cache=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{dataset}_{data}_yl',
                                        overwrite_cache=overwrite_cache, discrete_colors=True, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yl downstream-sample-class')
    def get_data(batch: LocalScanDownstreamEmbeddings):
        return batch.yl, batch.classes_g[:, label_id], None
    plotter.add_batches(run.iter_downstream_data_batches(dataset=dataset, data=data), get_data,
                        limit_samples=limit_samples)

    plotter.plot(f'yl_downstream_class_g_{dataset}',
                 colors=[[cm.viridis(0.)], [cm.viridis(1.)]], markers=['o', 'o'], labels=['Normal', 'Pneunomia'])


def plot_tsne_yg_downstream_sample_class(run: PretrainingRun, dataset, data='test', label_id=0, limit_samples=None,
                                         perplexity=30., overwrite_cache=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{dataset}_{data}_yg',
                                        overwrite_cache=overwrite_cache, discrete_colors=True, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yg downstream-sample-class')
    def get_data(batch: LocalScanDownstreamEmbeddings):
        return batch.yg, batch.classes_g[:, label_id], None
    plotter.add_batches(run.iter_downstream_data_batches(dataset=dataset, data=data), get_data,
                        limit_samples=limit_samples)

    plotter.plot(f'yg_downstream_class_g_{dataset}',
                 colors=[[cm.viridis(0.)], [cm.viridis(1.)]], markers=['o', 'o'], labels=['Normal', 'Pneunomia'])

def plot_tsne_yl_a(run: PretrainingRun, data='test', limit_samples=None,
                 perplexity=30., overwrite_cache=False, color_by_chexpert=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{data}_yl_a',
                                        overwrite_cache=overwrite_cache, discrete_colors=color_by_chexpert, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yl_a')
    def get_data(batch: ModelData):
        embeddings = batch.embeddings
        color = 'sample_id'
        if color_by_chexpert:
            color = chexpert_bin_labels_to_color_ids(batch.inputs.chexpert_bin_labels)
        return embeddings.yl_a, color, None
    plotter.add_batches(run.iter_model_data_batches(data=data, load_inputs=True, load_attentions=False), get_data,
                        limit_samples=limit_samples)

    if color_by_chexpert:
        plotter.plot(f'yl_a_chexpert',
                     colors=['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'],
                     markers=['x', 'o', 'o', 'o', 'o', 'o'],
                     labels=['Normal'] + chexpert_labels())
    else:
        plotter.plot(f'yl_a')

def plot_tsne_yl_b(run: PretrainingRun, data='test', limit_samples=None,
                 perplexity=30., overwrite_cache=False, color_by_chexpert=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{data}_yl_b',
                                        overwrite_cache=overwrite_cache, discrete_colors=color_by_chexpert, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yl_b')
    def get_data(batch: ModelData):
        embeddings = batch.embeddings
        color = 'sample_id'
        if color_by_chexpert:
            color = chexpert_bin_labels_to_color_ids(batch.inputs.chexpert_bin_labels)
        return embeddings.yl_b, color, embeddings.mask_b.binary_mask
    plotter.add_batches(run.iter_model_data_batches(data=data, load_inputs=True, load_attentions=False), get_data,
                        limit_samples=limit_samples)

    if color_by_chexpert:
        plotter.plot(f'yl_b_chexpert',
                     colors=['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'],
                     markers=['x', 'o', 'o', 'o', 'o', 'o'],
                     labels=['Normal'] + chexpert_labels())
    else:
        plotter.plot(f'yl_b')


def plot_tsne_yg_a(run: PretrainingRun, data='test', limit_samples=None,
                 perplexity=30., overwrite_cache=False, color_by_chexpert=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{data}_yg_a',
                                        overwrite_cache=overwrite_cache, discrete_colors=color_by_chexpert, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yg_a')
    def get_data(batch: ModelData):
        embeddings = batch.embeddings
        color = None
        if color_by_chexpert:
            color = chexpert_bin_labels_to_color_ids(batch.inputs.chexpert_bin_labels)
        return embeddings.yg_a, color, None
    plotter.add_batches(run.iter_model_data_batches(data=data, load_inputs=True, load_attentions=False), get_data,
                        limit_samples=limit_samples)
    if color_by_chexpert:
        plotter.plot(f'yg_a_chexpert',
                     colors=['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'],
                     markers=['x', 'o', 'o', 'o', 'o', 'o'],
                     labels=['Normal'] + chexpert_labels())
    else:
        plotter.plot(f'yg_a')


def plot_tsne_yg_b(run: PretrainingRun, data='test', limit_samples=None,
                 perplexity=30., overwrite_cache=False, color_by_chexpert=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{data}_yg_b',
                                        overwrite_cache=overwrite_cache, discrete_colors=color_by_chexpert, perplexity=perplexity)
    log.info('Loading data for plot t-SNE yg_b')
    def get_data(batch: ModelData):
        embeddings = batch.embeddings
        color = None
        if color_by_chexpert:
            color = chexpert_bin_labels_to_color_ids(batch.inputs.chexpert_bin_labels)
        return embeddings.yg_b, color, None
    plotter.add_batches(run.iter_model_data_batches(data=data, load_inputs=True, load_attentions=False), get_data,
                        limit_samples=limit_samples)
    if color_by_chexpert:
        plotter.plot(f'yg_b_chexpert',
                     colors=['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'],
                     markers=['x', 'o', 'o', 'o', 'o', 'o'],
                     labels=['Normal'] + chexpert_labels())
    else:
        plotter.plot(f'yg_b')


def chexpert_bin_labels_to_color_ids(chexpert_bin_labels):
    labels = chexpert_labels()
    B = chexpert_bin_labels[labels[0]].shape[0]
    color_ids = torch.zeros(B)
    for i, label in enumerate(labels):
        color_ids[chexpert_bin_labels[label] == 1] = i + 1
    return color_ids


def plot_tsne_zl(run: PretrainingRun, data='test', limit_samples=None,
                 perplexity=30., overwrite_cache=False, include_attended=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{data}_zl{"_att" if include_attended else ""}',
                                        overwrite_cache=overwrite_cache, discrete_colors=True, perplexity=perplexity)
    log.info('Loading data for plot t-SNE zl')
    def get_data(batch: ModelData):
        embeddings = batch.embeddings
        B, N_a, _ = embeddings.zl_a.size()
        _, N_b, _ = embeddings.zl_b.size()
        mask_a = torch.ones(B, N_a, dtype=bool)
        mask_b = embeddings.mask_b.binary_mask
        modality_a = torch.zeros(B, N_a, dtype=int)
        modality_b = torch.ones(B, N_b, dtype=int)
        modality_b2a = torch.full((B, N_a), 2, dtype=int)
        modality_a2b = torch.full((B, N_b), 3, dtype=int)
        if include_attended and embeddings.zl_b2a is not None and embeddings.zl_a2b is not None:
            return torch.cat([embeddings.zl_a, embeddings.zl_b, embeddings.zl_b2a, embeddings.zl_a2b], dim=1), \
                   torch.cat([modality_a, modality_b, modality_b2a, modality_a2b], dim=1), \
                   torch.cat([mask_a, mask_b, mask_a, mask_b], dim=1)
        elif include_attended and embeddings.zl_b2a is not None:
            return torch.cat([embeddings.zl_a, embeddings.zl_b, embeddings.zl_b2a], dim=1), \
                   torch.cat([modality_a, modality_b, modality_b2a], dim=1), \
                   torch.cat([mask_a, mask_b, mask_a], dim=1)
        elif include_attended and embeddings.zl_a2b is not None:
            return torch.cat([embeddings.zl_a, embeddings.zl_b, embeddings.zl_a2b], dim=1), \
                   torch.cat([modality_a, modality_b, modality_a2b], dim=1), \
                   torch.cat([mask_a, mask_b, mask_b], dim=1)
        else:
            return torch.cat([embeddings.zl_a, embeddings.zl_b], dim=1), \
                   torch.cat([modality_a, modality_b], dim=1), \
                   torch.cat([mask_a, embeddings.mask_b.binary_mask], dim=1)
    plotter.add_batches(run.iter_model_data_batches(data=data, load_inputs=False, load_attentions=False), get_data,
                        limit_samples=limit_samples)

    if include_attended:
        plotter.plot(f'zl',
                 colors=['tab:blue', 'tab:orange', 'tab:purple', 'tab:red'], markers=['o', 'o', 'x', 'x'],
                     labels=['Image regions', 'Report sentences', 'Report-to-image', 'Image-to-report'])
    else:
        plotter.plot(f'zl',
                 colors=['tab:blue', 'tab:orange'], markers=['o', 'o'], labels=['Image regions', 'Report sentences'])


def plot_tsne_zg(run: PretrainingRun, data='test', limit_samples=None,
                 perplexity=30., overwrite_cache=False):
    plotter = LocalEmbeddingTsnePlotter(run.run_path, embeddings_name=f'{data}_zg',
                                        overwrite_cache=overwrite_cache, discrete_colors=True, perplexity=perplexity)
    log.info('Loading data for plot t-SNE zg')
    def get_data(batch: ModelData):
        embeddings = batch.embeddings
        B, _ = embeddings.zg_a.size()
        modality_a = torch.zeros(B, dtype=int)
        modality_b = torch.ones(B, dtype=int)
        return torch.cat([embeddings.zg_a, embeddings.zg_b], dim=0), \
               torch.cat([modality_a, modality_b], dim=0), \
               None
    plotter.add_batches(run.iter_model_data_batches(data=data, load_inputs=False, load_attentions=False), get_data,
                        limit_samples=limit_samples)

    plotter.plot(f'zg',
                 colors=['tab:blue', 'tab:orange'], markers=['o', 'o'], labels=['Image', 'Report'])


def plot_embeddings(run):
    plot_tsne_yl_a(run, color_by_chexpert=True, limit_samples=100)
    plot_tsne_yl_b(run, color_by_chexpert=True, limit_samples=100)
    plot_tsne_yg_a(run, color_by_chexpert=True)
    plot_tsne_yg_b(run, color_by_chexpert=True)
    plot_tsne_zl(run, limit_samples=100, include_attended=True)
    plot_tsne_zg(run)


def plot_downstream_embeddings(run):
    plot_tsne_yl_downstream_class_overlap(run, 'rsna_seg', limit_samples=100)
    plot_tsne_yl_downstream_sample_class(run, 'rsna_seg', limit_samples=100)
    plot_tsne_yg_downstream_sample_class(run, 'rsna_seg')


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
