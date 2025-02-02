from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import path
from plot_utils import colors, markers
from plot_utils import load_benchmark_data


def _save_plot(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png')
    fig.savefig(save_path / f'{filename}.pdf')
    plt.close()


def _plot_grid(nc_mean, nc_std, acc_mean, acc_std, x, x_label, with_errorbars=False):

    def _plot_line(ax, x, y, label, marker, color=None, error=None):
        if color is None:
            if with_errorbars and error is not None:
                ax.errorbar(x, y, yerr=error ,label=label, marker=marker, markersize=5)
            else:
                ax.plot(x, y, label=label, marker=marker, markersize=5)
        else:
            if with_errorbars and error is not None:
                ax.errorbar(x, y, yerr=error ,label=label, marker=marker, markersize=5, color=color)
            else:
                ax.plot(x, y, label=label, marker=marker, markersize=5, color=color)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Subplot 00
    _plot_line(axes[0, 0], x, nc_mean['nc1_weak_between'], 'nc1_weak_between', markers[1], error=nc_std['nc1_weak_between'])
    _plot_line(axes[0, 0], x, nc_mean['nc1_weak_within'], 'nc1_weak_within', markers[2], error=nc_std['nc1_weak_within'])
    axes[0, 0].set_ylabel('nc1_weak')
    ax001 = axes[0, 0].twinx()
    _plot_line(ax001, x, nc_mean['nc1_cdnv'], 'nc1_cdnv', markers[3], color=colors[2], error=nc_std['nc1_cdnv'])
    ax001.set_ylabel('nc1_cdnv')
    #plot_line(ax001, x, nc['nc1_strong'], 'nc1_strong', markers[0], color=colors[3])

    # Subplot 01
    _plot_line(axes[0, 1], x, nc_mean['nc2_equinormness'], 'nc2_equinormness', markers[0], error=nc_std['nc2_equinormness'])
    axes[0, 1].set_ylabel('nc2_equinormness')
    ax011 = axes[0, 1].twinx()
    _plot_line(ax011, x, nc_mean['nc2_equiangularity'], 'nc2_equiangularity', markers[1], color=colors[1], error=nc_std['nc2_equiangularity'])
    ax011.set_ylabel('nc2_equiangularity')
    # plot_line(axes[0, 1], x, nc['gnc2_hyperspherical_uniformity'], 'gnc2_hyperspherical_uniformity', markers[2])

    # Subplot 10
    _plot_line(axes[1, 0], x, nc_mean['nc3_self_duality'], 'nc3_self_duality', markers[0], error=nc_std['nc3_self_duality'])
    ax101 = axes[1, 0].twinx()
    _plot_line(ax101, x, nc_mean['unc3_uniform_duality'], 'unc3_uniform_duality', markers[1], color=colors[1], error=nc_std['unc3_uniform_duality'])
    ax101.set_ylabel('unc3')
    axes[1, 0].set_ylabel('nc3')

    # Subplot 11
    _plot_line(axes[1, 1], x, nc_mean['nc4_classifier_agreement'], 'nc4_classifier_agreement', markers[0], error=nc_std['nc4_classifier_agreement'])
    _plot_line(axes[1, 1], x, acc_mean, 'acc', 'None', error=acc_std)
    axes[1, 1].set_ylabel('agreement/accuracy')

    # Legend subplot 00
    lines000, labels000 = axes[0, 0].get_legend_handles_labels()
    lines001, labels001 = ax001.get_legend_handles_labels()
    ax001.legend(lines000 + lines001, labels000 + labels001)
    # Legend subplot 01
    lines010, labels010 = axes[0, 1].get_legend_handles_labels()
    lines011, labels011 = ax011.get_legend_handles_labels()
    ax011.legend(lines010 + lines011, labels010 + labels011)
    # Legend subplot 10
    lines100, labels100 = axes[1, 0].get_legend_handles_labels()
    lines101, labels101 = ax101.get_legend_handles_labels()
    ax101.legend(lines100 + lines101, labels100 + labels101)
    # Legend subplot 11
    axes[1, 1].legend(loc=4)

    axes[0, 0].set_xlabel(x_label)
    axes[0, 1].set_xlabel(x_label)
    axes[1, 0].set_xlabel(x_label)
    axes[1, 1].set_xlabel(x_label)

    plt.tight_layout()
    return fig


def tolerant_mean(arrs):
    """https://stackoverflow.com/a/59281468"""
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = np.squeeze(l)
    mu = arr.mean(axis=-1)
    std = arr.std(axis=-1)
    return mu, std


def plot_nc(benchmark_name):
    _, epochs, acc, nc, _, _, _ = load_benchmark_data(benchmark_name)

    if benchmark_name == 'imagenet':
        with_errorbars = False
        nc_mean = nc
        nc_std = nc
        acc_mean = acc
        acc_std = acc
    
    else:
        # Split into runs
        idx = np.where(epochs == 1)[0][1:]
        epochs = np.split(epochs, indices_or_sections=idx)
        acc = np.split(acc, indices_or_sections=idx)

        for k, v in nc.items():
            nc[k] = np.split(v, indices_or_sections=idx)

        # Stats
        acc_mean, acc_std = tolerant_mean(acc)

        nc_mean = {}
        nc_std = {}
        for k, v in nc.items():
            nc_mean[k], nc_std[k] = tolerant_mean(v)
        
        epochs, _ = tolerant_mean(epochs)

        with_errorbars = True
    
    save_dir = path.res_plots / 'nc'
    save_dir.mkdir(exist_ok=True, parents=True)

    fig = _plot_grid(nc_mean, nc_std, acc_mean, acc_std, epochs, 'epoch', with_errorbars=with_errorbars)
    _save_plot(fig, save_dir, f'nc_{benchmark_name}_epoch')

    fig = _plot_grid(nc_mean, nc_std, acc_mean, acc_std, acc_mean, 'acc_val', with_errorbars=with_errorbars)
    _save_plot(fig, save_dir, f'nc_{benchmark_name}_acc')


if __name__ == '__main__':
    # plot_nc('cifar10')
    # plot_nc('cifar100')
    # plot_nc('imagenet200')
    # plot_nc('imagenet')
    plot_nc('alexnet')
    plot_nc('vgg')
    plot_nc('mobilenet')
