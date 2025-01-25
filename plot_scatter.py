from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np

import path
from plot_utils import benchmark2loaddirs, nc_metrics
from plot_utils import colors, metric_markers
from plot_utils import load_acc, load_nc_ood, numpify_dict, check_run_data


def trim_arrays(dict1, dict2, dict3, array1):
    min_length = min(
        min(arr.shape[0] for arr in dict1.values()),
        min(arr.shape[0] for arr in dict2.values())
    )
    
    trimmed_dict1 = {key: arr[:min_length] for key, arr in dict1.items()}
    trimmed_dict2 = {key: arr[:min_length] for key, arr in dict2.items()}
    trimmed_dict3 = {key: arr[:min_length] for key, arr in dict3.items()}
    
    trimmed_array1 = array1[:min_length]
    
    return trimmed_dict1, trimmed_dict2, trimmed_dict3, trimmed_array1


class IncompleteError(Exception):
    pass


def load_benchmark_data(benchmark_name,
                        ood_metric='AUROC',
                        ):
    # Get run dirs
    main_dirs = benchmark2loaddirs[benchmark_name]
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = natsorted([subdir for p in main_dirs if p.is_dir() for subdir in p.iterdir() if subdir.is_dir()], key=str)
    print('number of runs', len(run_dirs))

    save_dir = path.res_plots / main_dirs[0].relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    epochs = []
    run_ids = []
    acc = []
    nc = defaultdict(list)
    nood = []
    food = []

    # for run_dir in run_dirs:
    #     check_run_data(run_dir)

    for run_id, run_dir in enumerate(run_dirs):
        nc_dict, nearood_dict, farood_dict, epochs_ = load_nc_ood(run_dir, nc_split='val', ood_metric=ood_metric, benchmark=benchmark_name)
        acc_ = load_acc(run_dir, filter_epochs=epochs_, benchmark=benchmark_name)
        acc_ = list(acc_['val']['values'])

        # Needed if not all benchmarks are computed
        # nearood_dict, farood_dict, nc_dict, acc_ = trim_arrays(nearood_dict, farood_dict, nc_dict, acc_)

        near_lengths = [len(v) for v in nearood_dict.values()]
        far_lengths = [len(v) for v in farood_dict.values()]

        # Check for completeness
        if (not all([l0 == l1 for l0, l1 in zip(near_lengths, far_lengths)]) or 
            not all([l0 == near_lengths[0] for l0 in near_lengths]) or
            not all([l0 == far_lengths[0] for l0 in far_lengths])
        ):
            print('near', near_lengths)
            print('far',far_lengths)
            raise IncompleteError(run_dir)
        
        nearood = list(np.mean([v for v in nearood_dict.values()], axis=0))
        farood = list(np.mean([v for v in farood_dict.values()], axis=0))

        epochs.extend(epochs_)
        run_ids.extend([run_id for _ in range(len(epochs_))])
        acc.extend(acc_)
        nood.extend(nearood)
        food.extend(farood)

        for k, v in nc_dict.items():
            nc[k].extend(v)

    run_ids = np.array(run_ids)
    epochs = np.array(epochs)
    acc = np.array(acc)
    nc = numpify_dict(nc)
    nood = np.array(nood)
    food = np.array(food)

    return run_ids, epochs, acc, nc, nood, food, save_dir


def _plot(acc, nc, ood, run_ids, nc_metric, ood_metric, ood_label):
    assert len(acc) == len(nc) == len(ood), f'{len(acc)} {len(nc)} {len(ood)}'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # c = [colors[i] for i in run_ids]
    c = run_ids

    axes.ravel()[0].scatter(acc, nc, c=c, marker='o')
    axes.ravel()[0].set_xlabel(f'acc val')
    axes.ravel()[0].set_ylabel(nc_metric)

    axes.ravel()[1].scatter(acc, ood, c=c, marker='o')
    axes.ravel()[1].set_xlabel(f'acc val')
    axes.ravel()[1].set_ylabel(f'{ood_metric} {ood_label}')

    axes.ravel()[2].scatter(nc, ood, c=c, marker='o')
    axes.ravel()[2].set_xlabel(nc_metric)
    axes.ravel()[2].set_ylabel(f'{ood_metric} {ood_label}')

    plt.tight_layout()
    return fig


def _save(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    fig.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def _mean(a1, a2):
    x = np.vstack((a1, a2))
    return x.mean(axis=0)


def plot_scatter_all(benchmark_name,
                 ood_metric='AUROC',
                 ):
    print(f'plotting scatter {benchmark_name} {ood_metric} ...')
    run_ids, epochs, acc, nc, nearood, farood, save_dir = load_benchmark_data(benchmark_name, ood_metric)

    # Epochs to color index
    _, color_id = np.unique(epochs, return_inverse=True)
    color_id += 1

    save_dir = path.res_plots / 'scatter' / benchmark_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for nc_metric in nc_metrics:
        print(nc_metric)
        # Plot and save
        fig = _plot(acc, nc[nc_metric], _mean(nearood, farood), color_id, nc_metric, ood_metric, 'mean')
        _save(fig, save_dir, f'mean_{nc_metric}_{ood_metric}')
        fig = _plot(acc, nc[nc_metric], nearood, color_id, nc_metric, ood_metric, 'near')
        _save(fig, save_dir, f'scatter_near_{nc_metric}_{ood_metric}')
        fig = _plot(acc, nc[nc_metric], farood, color_id, nc_metric, ood_metric, 'far')
        _save(fig, save_dir, f'scatter_far_{nc_metric}_{ood_metric}')


if __name__ == '__main__':
    # plot_scatter_all('cifar10')
    plot_scatter_all('cifar100')
    # plot_scatter_all('imagenet200')
    # plot_scatter_all('imagenet')
    # plot_scatter_all('alexnet')
    # plot_scatter_all('mobilenet')
    # plot_scatter_all('vgg')