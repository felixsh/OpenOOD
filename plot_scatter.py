import matplotlib.pyplot as plt
import numpy as np

import path
from plot_utils import load_benchmark_data, mean_ood_2dict
from plot_utils import nc_metrics_cov, nc_metrics_mean, ood_methods


def _plot(acc, nc, ood, color_id, nc_metric, ood_metric, ood_label):
    assert len(acc) == len(nc) == len(ood), f'{len(acc)} {len(nc)} {len(ood)}'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes.ravel()[0].scatter(acc, nc, c=color_id, marker='o')
    axes.ravel()[0].set_xlabel(f'acc val')
    axes.ravel()[0].set_ylabel(nc_metric)

    axes.ravel()[1].scatter(acc, ood, c=color_id, marker='o')
    axes.ravel()[1].set_xlabel(f'acc val')
    axes.ravel()[1].set_ylabel(f'{ood_metric} {ood_label}')

    axes.ravel()[2].scatter(nc, ood, c=color_id, marker='o')
    axes.ravel()[2].set_xlabel(nc_metric)
    axes.ravel()[2].set_ylabel(f'{ood_metric} {ood_label}')

    plt.tight_layout()
    return fig


def _save(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    fig.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_scatter_all(benchmark_name, nc_split='val', ood_metric='AUROC', reduction='mean'):
    print(f'plotting scatter {benchmark_name} {ood_metric} ...')
    _, epochs, acc, nc_dict, nood_dict, food_dict, save_dir = load_benchmark_data(benchmark_name, nc_split, ood_metric)

    # Mean over ood methods
    ood_values = mean_ood_2dict(nood_dict, food_dict)

    # Epochs to color index
    _, color_id = np.unique(epochs, return_inverse=True)
    color_id += 1

    if reduction == 'mean':
        nc_metrics = nc_metrics_mean
    elif reduction == 'cov':
        nc_metrics = nc_metrics_cov
    else:
        raise NotImplementedError

    save_dir = path.res_plots / 'scatter' / benchmark_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for nc_metric in nc_metrics:
        print(nc_metric)
        nc_values = nc_dict[nc_metric]
        # Plot and save
        fig = _plot(acc, nc_values, ood_values, color_id, nc_metric, ood_metric, 'mean')
        _save(fig, save_dir, f'scatter_mean_{nc_metric}_{ood_metric}')
        # fig = _plot(acc, nc_values, nood_values, color_id, nc_metric, ood_metric, 'near')
        # _save(fig, save_dir, f'scatter_near_{nc_metric}_{ood_metric}')
        # fig = _plot(acc, nc_values, food_values, color_id, nc_metric, ood_metric, 'far')
        # _save(fig, save_dir, f'scatter_far_{nc_metric}_{ood_metric}')


def _scatter(ax, x, y, c, label=None, remove_outlier=True):
    x = np.squeeze(x)
    y = np.squeeze(y)

    if remove_outlier:
        leave_out = 10
        idx = np.argsort(x)
        direction = 1 if x[0] < x[-1] else -1

        idx = idx[::direction][leave_out:]
        x = x[idx]
        y = y[idx]
        if not isinstance(c, str):
            c = c[idx]

        # Lighter colors more on top
        if not isinstance(c, str):
            idx = np.argsort(c)
            x = x[idx]
            y = y[idx]
            c = c[idx]

    ax.scatter(x, y, c=c, marker='o', label=label, alpha=0.5)


def plot_scatter_tableau(benchmark_name, nc_split='val', ood_metric='AUROC', reduction='mean'):
    print(f'plotting scatter {benchmark_name} nc_{nc_split} ...')
    _, epochs, acc_val, acc_train, nc_dict, nood_dict, food_dict, save_dir = load_benchmark_data(benchmark_name, nc_split, ood_metric)

    # Mean over ood methods
    ood_values = mean_ood_2dict(nood_dict, food_dict)

    # Epochs to color index
    _, color_id = np.unique(epochs, return_inverse=True)
    color_id += 1

    if reduction == 'mean':
        nc_metrics = nc_metrics_mean
    elif reduction == 'cov':
        nc_metrics = nc_metrics_cov
    else:
        raise NotImplementedError

    save_dir = path.res_plots / 'scatter_tableau'
    save_dir.mkdir(parents=True, exist_ok=True)

    # assert len(nc_dict.keys()) == len(nc_metrics)

    # nc_dict['acc val'] = acc

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    ax = axes.ravel()

    assert len(acc_train) == len(acc_val) == len(ood_values), f'{len(acc_train)} {len(acc_val)} {len(ood_values)}'
 
    _scatter(ax[0], acc_val, ood_values, c=color_id)
    ax[0].set_xlabel('accuracy val')
    ax[0].set_ylabel(ood_metric)
    _scatter(ax[1], acc_train, ood_values, c=color_id)
    ax[1].set_xlabel('accuracy train')
    ax[1].set_ylabel(ood_metric)
    for a, key in zip(ax[2:], nc_metrics):
        _scatter(a, nc_dict[key], ood_values, c=color_id)
        a.set_xlabel(key)
        a.set_ylabel(ood_metric)
    
    ax[-1].axis('off')

    plt.tight_layout()

    _save(fig, save_dir, f'scatter_tableau_{benchmark_name}_{reduction}')


def plot_scatter_tableau_single(benchmark_name, nc_split='val', ood_metric='AUROC', reduction='mean'):
    print(f'plotting scatter {benchmark_name} nc_{nc_split} ...')
    _, epochs, acc_val, acc_train, nc_dict, nood_dict, food_dict, save_dir = load_benchmark_data(benchmark_name, nc_split, ood_metric)

    if reduction == 'mean':
        nc_metrics = nc_metrics_mean
    elif reduction == 'cov':
        nc_metrics = nc_metrics_cov
    else:
        raise NotImplementedError

    save_dir = path.res_plots / 'scatter_tableau_single'
    save_dir.mkdir(parents=True, exist_ok=True)

    # assert len(nc_dict.keys()) == len(nc_metrics)

    # nc_dict['acc val'] = acc

    for ood_method in ood_methods:
        ood_method = ood_method[1:]
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        ax = axes.ravel()

        _scatter(ax[0], acc_val, food_dict[ood_method], c='tab:blue', label='far')
        _scatter(ax[0], acc_val, nood_dict[ood_method], c='tab:orange', label='near')
        ax[0].set_xlabel('accuracy val')
        ax[0].set_ylabel(ood_metric)
        ax[0].legend()

        _scatter(ax[1], acc_train, food_dict[ood_method], c='tab:blue')
        _scatter(ax[1], acc_train, nood_dict[ood_method], c='tab:orange')
        ax[1].set_xlabel('accuracy train')
        ax[1].set_ylabel(ood_metric)

        for a, key in zip(ax[2:], nc_metrics):
            _scatter(a, nc_dict[key], food_dict[ood_method], c='tab:blue')
            _scatter(a, nc_dict[key], nood_dict[ood_method], c='tab:orange')
            a.set_xlabel(key)
            a.set_ylabel(ood_metric)

        ax[-1].axis('off')

        plt.suptitle(f'{benchmark_name} {ood_method}')

        plt.tight_layout()

        _save(fig, save_dir, f'scatter_tableau_single_{benchmark_name}_{reduction}_{ood_method}')


if __name__ == '__main__':
    # plot_scatter_all('cifar10')
    # plot_scatter_all('cifar100')
    # plot_scatter_all('imagenet200')
    # plot_scatter_all('imagenet')
    # plot_scatter_all('alexnet')
    # plot_scatter_all('mobilenet')
    # plot_scatter_all('vgg')

    # plot_scatter_tableau('cifar10', reduction='mean')
    # plot_scatter_tableau('cifar100', reduction='mean')
    # plot_scatter_tableau('imagenet200', reduction='mean')
    # plot_scatter_tableau('imagenet', reduction='mean')
    # plot_scatter_tableau('alexnet', reduction='mean')
    # plot_scatter_tableau('mobilenet', reduction='mean')
    # plot_scatter_tableau('vgg', reduction='mean')

    # plot_scatter_tableau('cifar10', reduction='cov')
    # plot_scatter_tableau('cifar100', reduction='cov')
    # plot_scatter_tableau('imagenet200', reduction='cov')
    # plot_scatter_tableau('imagenet', reduction='cov')
    # plot_scatter_tableau('alexnet', reduction='cov')
    # plot_scatter_tableau('mobilenet', reduction='cov')
    # plot_scatter_tableau('vgg', reduction='cov')

    plot_scatter_tableau_single('cifar10', reduction='mean')
    plot_scatter_tableau_single('imagenet200', reduction='mean')
