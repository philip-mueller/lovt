import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis.data_exporter import get_paper_data, load_downstream_results, RESULTS_PATH


def plot_downstream_results(downstream_task, metric, ylabel, category=None):
    data = get_paper_data(category=category).join(load_downstream_results(downstream_task))

    colors = ['tab:red' if is_baseline else 'tab:blue' for is_baseline in data['baseline']]
    run_paper_names = data['paper_name']
    x_pos = np.arange(len(run_paper_names))
    means = data[f'{metric}_mean']
    conf_intervals = data[f'{metric}_95_interval']

    fig, ax = plt.subplots()
    ax.bar(x_pos, 100 * means, yerr=100 * conf_intervals, color=colors, align='center', alpha=0.5, ecolor='black', capsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_paper_names, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
    ax.yaxis.grid(True)

    baseline_patch = mpatches.Patch(color='tab:red', alpha=0.5, label='Baseline')
    ours_patch = mpatches.Patch(color='tab:blue', alpha=0.5, label='Ours')
    ax.legend(handles=[baseline_patch, ours_patch])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'generated', f'downstream_{downstream_task}_{metric.replace("/", "_")}.pdf'))
