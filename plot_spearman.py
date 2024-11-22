from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  

import path
from plot_scatter import benchmark2loaddirs
from plot_utils import load_acc, load_nc, load_ood


def trim_arrays(dict1, dict2, dict3, array1):
    min_length = min(
        min(arr.shape[0] for arr in dict1.values()),
        min(arr.shape[0] for arr in dict2.values()),
        min(arr.shape[0] for arr in dict3.values()),
    )
    
    trimmed_dict1 = {key: arr[:min_length] for key, arr in dict1.items()}
    trimmed_dict2 = {key: arr[:min_length] for key, arr in dict2.items()}
    trimmed_dict3 = {key: arr[:min_length] for key, arr in dict3.items()}
    
    trimmed_array1 = array1[:min_length]
    
    return trimmed_dict1, trimmed_dict2, trimmed_dict3, trimmed_array1


def load_benchmark_data(benchmark_name,
                        ood_metric='AUROC',
                        ):
    # Get run dirs
    main_dirs = benchmark2loaddirs[benchmark_name]
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = [subdir for p in main_dirs if p.is_dir() for subdir in p.iterdir() if subdir.is_dir()]
    save_dir = path.res_plots / main_dirs[0].relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    nc_dict = defaultdict(list)
    acc_list = []
    nearood_list = []
    farood_list = []

    for run_dir in run_dirs:
        nc, _ = load_nc(run_dir)
        nearood_dict, farood_dict, epochs_ = load_ood(run_dir, ood_metric=ood_metric)
        acc = load_acc(run_dir, filter_epochs=epochs_)

        nc = {k: np.squeeze(v) for k, v in nc.items()}
        acc = acc['val']['values']

        # Needed if not all benchmarks are computed
        nearood_dict, farood_dict, nc, acc = trim_arrays(nearood_dict, farood_dict, nc, acc)
        
        nearood = np.mean([v for v in nearood_dict.values()], axis=0)
        farood = np.mean([v for v in farood_dict.values()], axis=0)

        acc_list.extend(acc)
        nearood_list.extend(nearood)
        farood_list.extend(farood)

        for k, v in nc.items():
            nc_dict[k].extend(v)

    nc_dict = {k: np.array(v) for k, v in nc_dict.items()}
    
    nc_dict['accuracy'] = np.array(acc_list)
    nc_dict['nearood'] = np.array(nearood_list)
    nc_dict['farood'] = np.array(farood_list)

    return nc_dict, save_dir


def _plot(data_dict):
    df = pd.DataFrame.from_dict(data_dict)

    spearman_corr = df.corr(method='spearman')
    
    # Create a mask for the lower triangle (including the diagonal)
    mask = np.tril(np.ones_like(spearman_corr, dtype=bool), k=-1)
    
    fig = plt.figure(figsize=(10, 8))
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        spearman_corr,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        mask=mask,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
        annot_kws={"size": 8, "color": "black"},
        linewidths=.5
    )
    
    plt.tight_layout()
    return fig


def _save(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    fig.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_spearman(benchmark_name,
                  ood_metric='AUROC',
                  ):
    data_dict, save_dir = load_benchmark_data(benchmark_name, ood_metric)
    fig = _plot(data_dict)
    _save(fig, save_dir, f'spearman_{ood_metric}')


if __name__ == '__main__':
    plot_spearman('cifar10')
    #plot_spearman('cifar100')
    plot_spearman('imagenet200')
