import math
import os

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from analysis.data_exporter import get_paper_data, load_analysis_results, RESULTS_PATH, \
    get_run_definitions_df
from analysis.postprocess_run import PretrainingRun
from metrics.embedding_metrics import prepare_mask
from models.pretraining.pretraining_utils import EmbeddingsData


def l_emb_for_modality(modality):
    return r'$\mathbf{y}^{\mathcal{' + modality.upper() + '}}$'

def g_emb_for_modality(modality):
    return r'$\mathbf{\bar{y}}^{\mathcal{' + modality.upper() + '}}$'


def calculate_rows(df, emb_l='yl_a', emb_g='yg_a', d_l=2048, d_g=2048):
    # for std normalization see https://arxiv.org/pdf/2011.10566.pdf
    l_std = df[f'emb/{emb_l}/ch_std']
    l_std_centroid = df[f'emb/{emb_l}/ch_std_sample_centroids']
    result_df = pd.DataFrame()
    result_df['paper_name'] = df['paper_name']
    result_df['has_local'] = df['has_local']
    result_df['has_global'] = df['has_global']
    result_df['l_std'] = l_std * math.sqrt(d_l)
    result_df['l_std_sample'] = df[f'emb/{emb_l}/ch_std_per_sample'] * math.sqrt(d_l)
    result_df['l_std_sample__std'] = df[f'emb/{emb_l}/ch_std_per_sample__std'] * math.sqrt(d_l)
    result_df['l_std_centroid'] = l_std_centroid * math.sqrt(d_l)
    result_df['l_std_ratio'] = l_std_centroid / l_std
    result_df['l_uni_sample'] = df[f'emb/{emb_l}/per_sample_uniformity']
    result_df['l_uni_sample__std'] = df[f'emb/{emb_l}/per_sample_uniformity__std']
    result_df['l_cov'] = df[f'emb/{emb_l}/cov_offdiagonal__batch_estimate']
    result_df['l_cov_sample'] = df[f'emb/{emb_l}/cov_offdiagonal_per_sample']
    result_df['l_cov_sample__std'] = df[f'emb/{emb_l}/cov_offdiagonal_per_sample__std']
    result_df['l_cov_centroid'] = df[f'emb/{emb_l}/cov_offdiagonal_sample_centroids']
    result_df['g_std'] = df[f'emb/{emb_g}/ch_std'] * math.sqrt(d_g)
    result_df['g_uni'] = df[f'emb/{emb_g}/uniformity']
    result_df['g_cov'] = df[f'emb/{emb_g}/cov_offdiagonal']
    result_df['order'] = df['order']
    result_df = result_df.sort_values('order')

    return result_df


def format_rows(df):
    out_df = pd.DataFrame()
    out_df['paper_name'] = df['paper_name']
    out_df['l_std'] = df['l_std'].apply('{:.2f}'.format)
    out_df['l_std_sample'] = "$" + df['l_std_sample'].apply('{:.2f}'.format) \
                             + r" \pm " + df['l_std_sample__std'].apply('{:.2f}'.format) + "$"
    out_df['l_std_centroid'] = df['l_std_centroid'].apply('{:.2f}'.format)
    out_df['l_std_ratio'] = df['l_std_ratio'].apply('{:.2f}'.format)
    out_df['l_uni_sample'] = "$" + df['l_uni_sample'].apply('{:.2f}'.format) \
                             + r" \pm " + df['l_uni_sample__std'].apply('{:.2f}'.format) + "$"
    out_df['l_cov'] = df['l_cov'].apply('{:.1e}'.format)
    out_df['l_cov_sample'] = df['l_cov_sample'].apply('{:.1e}'.format)
    out_df['l_cov_centroid'] = df['l_cov_centroid'].apply('{:.1e}'.format)
    out_df['g_std'] = df['g_std'].apply('{:.2f}'.format)
    out_df['g_uni'] = df['g_uni'].apply('{:.2f}'.format)
    out_df['g_cov'] = df['g_cov'].apply('{:.1e}'.format)
    out_df = out_df.applymap(lambda val: '' if 'nan' in val else val)
    return out_df


def write_tex_table(formatted_df: pd.DataFrame, name, modality):
    header = r"""\begin{tabular}{lccccccccccc}
  \toprule
  Model 
    & \multicolumn{4}{c}{$\bm{y}_""" + modality + r"""$ channel std} 
    & $\bm{y}_""" + modality + r"""$ uniformity 
    & \multicolumn{3}{c}{$\bm{y}_""" + modality + r"""$off-diagonal covariance} 
    & $\bar{\bm{y}}_""" + modality + r"""$ ch. std 
    & $\bar{\bm{y}}_""" + modality + r"""$ unif. 
    & $\bar{\bm{y}}_""" + modality + r"""$ cov. \\
  & total & sample & centroid & ratio
    & sample 
    & total & sample & centroid 
    &&& \\
  \midrule \\
"""

    body = '\n'.join('  ' + ' & '.join(col for col in row) + r' \\' for row in formatted_df.values.tolist())

    footer = r"""
  \bottomrule
\end{tabular}
"""

    with open(os.path.join(RESULTS_PATH, 'generated', f'emb_analysis_table_{name}.tex'), 'w') as f:
        f.write(header + body + footer)


def compute_emb_properties_table(df, name='y_a', modality='s'):
    write_tex_table(format_rows(df), name=name, modality=modality)


def plot_uniformity(df, name, modality='s'):
    labels = df['paper_name']
    x_pos = np.arange(len(labels))
    unif = -df[f'l_uni_sample']
    unif_std = df[f'l_uni_sample__std']
    g_unif = -df[f'g_uni']

    fig, ax = plt.subplots()

    ax.bar(x_pos - 0.15, unif, width=0.3, yerr=unif_std, label=f'Mean per-sample uniformity {l_emb_for_modality(modality)}',
           alpha=0.5, ecolor='black', capsize=5, color='tab:orange')
    ax.bar(x_pos + 0.15, g_unif, width=0.3, label=f'Uniformity {g_emb_for_modality(modality)}', alpha=0.5, color='tab:red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    #ax.xaxis.tick_top()
    ax.set_ylabel(r'Uniformity')
    ax.set_ymargin(0.1)
    ax.legend(loc='upper center', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'generated', f'uniformity_{name}.pdf'), bbox_inches="tight")
    #plt.show()


def plot_cov(df, name, modality='s'):
    labels = df['paper_name']
    x_pos = np.arange(len(labels))
    cov = df[f'l_cov']
    cov_sample = df[f'l_cov_sample']
    cov_sample__std = df[f'l_cov_sample__std']
    cov_centroid = df[f'l_cov_centroid']
    g_cov = df[f'g_cov']

    fig, ax = plt.subplots()
    width = 0.6 / 4
    ax.bar(x_pos - 3 * width / 2, cov, width=width, label=f'Cov {l_emb_for_modality(modality)}', alpha=0.5)
    ax.bar(x_pos - width / 2, cov_sample, width=width, yerr=cov_sample__std, label=f'Mean per-sample cov {l_emb_for_modality(modality)}',
           alpha=0.5, ecolor='black', capsize=3)
    ax.bar(x_pos + width / 2, cov_centroid, width=width, label=f'Cov of per-sample centroids {l_emb_for_modality(modality)}', alpha=0.5)
    ax.bar(x_pos + 3 * width / 2, g_cov, width=width, label=f'Cov {g_emb_for_modality(modality)}', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    ax.set_yscale('log')
    ax.set_ylabel(r'Sum of squared off-diagonal elements of covariance')
    ax.set_ymargin(0.2)
    ax.legend(loc='upper center', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'generated', f'covariance_{name}.pdf'), bbox_inches="tight")
    #plt.show()


def plot_variances(df, name, modality='s'):
    labels = df['paper_name']
    x_pos = np.arange(len(labels))
    #var = df[f'l_std'] ** 2
    #centroid_var = df[f'l_std_centroid'] ** 2
    #g_var = df[f'g_std'] ** 2
    std = df[f'l_std']
    std_sample = df[f'l_std_sample']
    std_sample__std = df[f'l_std_sample__std']
    std_centroid = df[f'l_std_centroid']
    g_std = df[f'g_std']

    fig, ax = plt.subplots()

    #ax.bar(x_pos - 0.15, centroid_var, width=0.3,
    #       label=f'Variance of per-sample centroids {l_emb_for_modality(modality)}', alpha=0.5)
    #ax.bar(x_pos - 0.15, var - centroid_var, width=0.3, bottom=centroid_var,
    #       label=f'Mean per-sample variance {l_emb_for_modality(modality)}', alpha=0.5)
    #ax.bar(x_pos + 0.15, g_var, width=0.3, label=f'Variance {g_emb_for_modality(modality)}', alpha=0.5)

    width = 0.6 / 4
    ax.bar(x_pos - 3 * width / 2, std, width=width, label=f'Std {l_emb_for_modality(modality)}', alpha=0.5)
    ax.bar(x_pos - width / 2, std_sample, width=width, yerr=std_sample__std,
           label=f'Mean per-sample std {l_emb_for_modality(modality)}',
           alpha=0.5, ecolor='black', capsize=3)
    ax.bar(x_pos + width / 2, std_centroid, width=width,
           label=f'Std of per-sample centroids {l_emb_for_modality(modality)}', alpha=0.5)
    ax.bar(x_pos + 3 * width / 2, g_std, width=width, label=f'Std {g_emb_for_modality(modality)}', alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    d_text = 'd^{\mathcal{' + modality.upper() + '}}'
    ax.set_yticklabels([r'$\frac{1}{4 \sqrt{' + d_text + '}}$', r'$\frac{1}{2 \sqrt{' + d_text + '}}$',
                        r'$\frac{3}{4 \sqrt{' + d_text + '}}$', r'$\frac{1}{\sqrt{' + d_text + '}}$'])
    ax.set_ylabel(r'Mean per-channel std')
    ax.legend(loc='upper center', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'generated', f'std_{name}.pdf'), bbox_inches="tight")
    #plt.show()


def plot_modality_alignment(df):
    # Note: mean of per-sample alignment
    labels = df['paper_name']
    assign_s2r = 1 - df['emb/zl_a||zl_b/assignment_1to1']
    assign_r2s = 1 - df['emb/zl_a||zl_b/assignment_2to2']
    assignment_score = assign_r2s + assign_s2r
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots()

    ax.barh(y_pos, assign_s2r, height=0.6, label=r'Image regions ($\mathbf{z}^\mathcal{I}$)', alpha=0.5)
    ax.barh(y_pos, assign_r2s, height=0.6, left=1 - assign_r2s, label=r'Report sentences ($\mathbf{z}^\mathcal{R}$)', alpha=0.5)
    ax.vlines(0.5, -0.6, len(labels) - 0.4 + 0.6, colors='black', alpha=0.3, linestyle='--')
    for i, score in enumerate(assignment_score):
        ax.text(0.5, i, '{:.0f} %'.format(score * 100), horizontalalignment='center',
                verticalalignment='center')#, transform=ax.transAxes)

    ax.set_xlim(0., 1.)
    ax.set_ylim(-0.6, len(labels) - 0.4 + 0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    ax.set_xticks([0., 0.5, 1.0])
    ax.set_xticklabels(['Image', 'Indistinguishable', 'Report'])
    ax.legend(loc='upper center', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'generated', f'local_modality_assignment.pdf'), bbox_inches="tight")
    #plt.show()


def plot_alignment(run_definitions, normalized=True):
    run_definitions = run_definitions[~run_definitions.isna()['run_folder']]
    runs_l2_a = []
    runs_l2_b = []
    runs_l2_g = []
    run_names = []
    paper_names = []

    for name, run_def in tqdm(run_definitions.iterrows()):
        run_folder = run_def['run_folder']
        if len(run_folder) == 0:
            continue
        try:
            run = PretrainingRun.from_run_path(run_folder)
        except AssertionError:
            continue
        l2_a = []
        l2_b = []
        l2_g = []
        for batch in run.iter_model_data_batches(data='test', load_inputs=False, load_attentions=False):
            embeddings: EmbeddingsData = batch.embeddings
            if normalized:
                embeddings = embeddings.normalize()
            mask_a = prepare_mask(embeddings.zl_a, embeddings.mask_a)
            if embeddings.zl_b2a is not None:
                l2_a.append(F.pairwise_distance(embeddings.zl_a[mask_a], embeddings.zl_b2a[mask_a]))
            mask_b = prepare_mask(embeddings.zl_b, embeddings.mask_b)
            if embeddings.zl_a2b is not None:
                l2_b.append(F.pairwise_distance(embeddings.zl_b[mask_b], embeddings.zl_a2b[mask_b]))
            if embeddings.zg_a is not None and embeddings.zg_b is not None:
                l2_g.append(F.pairwise_distance(embeddings.zg_a, embeddings.zg_b))
        run_names.append(name)
        paper_names.append(run_def['paper_name'])
        runs_l2_a.append(torch.cat(l2_a, dim=0).numpy() if len(l2_a) > 0 else np.nan)
        runs_l2_b.append(torch.cat(l2_b, dim=0).numpy() if len(l2_b) > 0 else np.nan)
        runs_l2_g.append(torch.cat(l2_g, dim=0).numpy() if len(l2_g) > 0 else np.nan)

    if len(run_names) == 0:
        return

    df = pd.DataFrame({
        'name': run_names,
        'paper_name': paper_names,
        'l2_a': runs_l2_a,
        'l2_b': runs_l2_b,
        'l2_g': runs_l2_g
    })
    df = df.set_index('name')

    name_postfix = '_normalized' if normalized else ''
    plt.tight_layout()
    # global
    fig, ax = plt.subplots()
    positions = np.arange(df.shape[0])
    plot_boxplot(ax, df['l2_g'], pos=positions, width=0.2, c='tab:gray')
    #ax.set_xlim(-0.5, len(run_names))
    ax.set_xticks(positions)
    ax.set_xticklabels(df['paper_name'], rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    global_legend = mpatches.Patch(color='tab:gray', alpha=0.5, label='Global')
    ax.legend(handles=[global_legend], loc='upper center', fontsize='small')
    ax.set_ylabel(r'Alignment quality ($\ell_2$-distance)')
    fig.savefig(os.path.join(RESULTS_PATH, 'generated', f'global_alignment{name_postfix}.pdf'), bbox_inches="tight")

    # local
    fig, ax = plt.subplots()
    df = df[df.l2_a.notnull() | df.l2_b.notnull()]
    df['positions'] = np.arange(df.shape[0])
    df_a = df[df.l2_a.notnull()]
    plot_boxplot(ax, df_a['l2_a'], pos=df_a['positions'] - 0.20, width=0.3, c='tab:blue')
    df_b = df[df.l2_b.notnull()]
    plot_boxplot(ax, df_b['l2_b'], pos=df_b['positions'] + 0.20, width=0.3, c='tab:orange')
    #ax.set_xlim(-0.5, df.count())
    ax.set_ymargin(0.2)
    ax.set_xticks(df['positions'])
    ax.set_xticklabels(df['paper_name'], rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    scan_legend = mpatches.Patch(color='tab:blue', alpha=0.5, label='Report-to-image')
    report_legend = mpatches.Patch(color='tab:orange', alpha=0.5, label='Image-to-report')
    ax.legend(handles=[scan_legend, report_legend], loc='upper center', fontsize='small', ncol=2)
    ax.set_ylabel(r'Alignment quality ($\ell_2$-distance)')
    fig.savefig(os.path.join(RESULTS_PATH, 'generated', f'local_alignment{name_postfix}.pdf'), bbox_inches="tight")


def plot_boxplot(ax, data, pos, width, c):
    ax.boxplot(data, positions=pos, widths=width,
               patch_artist=True,
               boxprops=dict(color='black', facecolor=c, alpha=0.5),
               capprops=dict(color='black', alpha=0.5),
               whiskerprops=dict(color='black', alpha=0.5),
               flierprops=dict(markeredgecolor=c, markerfacecolor=c, alpha=0.5),
               medianprops=dict(color='black', alpha=0.5))


def plot_emb_properties(category=None, skip_alignment=False):
    df = get_paper_data(category=category).join(get_run_definitions_df()).sort_values('order')
    if not skip_alignment:
        plot_alignment(df, normalized=True)
        plot_alignment(df, normalized=False)

    df = get_paper_data(category=category).join(load_analysis_results()).sort_values('order')
    df_a = calculate_rows(df, emb_l='yl_a', emb_g='yg_a', d_l=2048, d_g=2048)
    df_b = calculate_rows(df, emb_l='yl_b', emb_g='yg_b', d_l=768, d_g=768)

    compute_emb_properties_table(df_a, name='y_a', modality='i')
    compute_emb_properties_table(df_b, name='y_b', modality='r')

    plot_variances(df_a, name='y_a', modality='i')
    plot_variances(df_b, name='y_b', modality='r')
    plot_uniformity(df_a, name='y_a', modality='i')
    plot_uniformity(df_b, name='y_b', modality='r')
    plot_cov(df_a, name='y_a', modality='i')
    plot_cov(df_b, name='y_b', modality='r')
    plot_modality_alignment(df)
