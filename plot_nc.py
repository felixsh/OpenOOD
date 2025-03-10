import matplotlib.pyplot as plt
import numpy as np

import path
from plot_utils import colors, load_benchmark_data, markers


def _save_plot(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png')
    fig.savefig(save_path / f'{filename}.pdf')
    plt.close()


def _plot_grid(
    nc_val_mean,
    nc_val_std,
    acc_val_mean,
    acc_val_std,
    acc_train_mean,
    acc_train_std,
    x,
    x_label,
    nc_train_mean=None,
    nc_train_std=None,
    with_errorbars=False,
):
    def _plot_line(
        ax, x, y, label=None, marker=None, color=None, linestyle='-', error=None
    ):
        if linestyle != '-':
            label = None

        if color is None:
            if with_errorbars and error is not None:
                ax.errorbar(
                    x,
                    y,
                    yerr=error,
                    label=label,
                    marker=marker,
                    markersize=5,
                    linestyle=linestyle,
                )
            else:
                ax.plot(
                    x, y, label=label, marker=marker, markersize=5, linestyle=linestyle
                )
        else:
            if with_errorbars and error is not None:
                ax.errorbar(
                    x,
                    y,
                    yerr=error,
                    label=label,
                    marker=marker,
                    markersize=5,
                    color=color,
                    linestyle=linestyle,
                )
            else:
                ax.plot(
                    x,
                    y,
                    label=label,
                    marker=marker,
                    markersize=5,
                    color=color,
                    linestyle=linestyle,
                )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax in axes.flatten():
        ax.set_xscale('log')

    # Subplot 00 ======================================================================================
    _plot_line(
        axes[0, 0],
        x,
        nc_val_mean['nc1_weak_between'],
        'nc1_weak_between',
        markers[1],
        error=nc_val_std['nc1_weak_between'],
        color=colors[0],
    )
    if nc_train_mean is not None:
        _plot_line(
            axes[0, 0],
            x,
            nc_train_mean['nc1_weak_between'],
            linestyle='--',
            color=colors[0],
        )

    _plot_line(
        axes[0, 0],
        x,
        nc_val_mean['nc1_weak_within'],
        'nc1_weak_within',
        markers[2],
        error=nc_val_std['nc1_weak_within'],
        color=colors[1],
    )
    if nc_train_mean is not None:
        _plot_line(
            axes[0, 0],
            x,
            nc_train_mean['nc1_weak_within'],
            linestyle='--',
            color=colors[1],
        )

    axes[0, 0].set_ylabel('nc1_weak')
    ax001 = axes[0, 0].twinx()
    _plot_line(
        ax001,
        x,
        nc_val_mean['nc1_cdnv_mean'],
        'nc1_cdnv_mean',
        markers[3],
        color=colors[2],
        error=nc_val_std['nc1_cdnv_mean'],
    )
    if nc_train_mean is not None:
        _plot_line(
            ax001,
            x,
            nc_train_mean['nc1_cdnv_mean'],
            linestyle='--',
            color=colors[2],
        )
    ax001.set_ylabel('nc1_cdnv_mean')
    # plot_line(ax001, x, nc['nc1_strong'], 'nc1_strong', markers[0], color=colors[3])

    # Subplot 01 ======================================================================================
    _plot_line(
        axes[0, 1],
        x,
        nc_val_mean['nc2_equinormness_mean'],
        'nc2_equinormness_mean',
        markers[0],
        error=nc_val_std['nc2_equinormness_mean'],
        color=colors[0],
    )
    if nc_train_mean is not None:
        _plot_line(
            axes[0, 1],
            x,
            nc_train_mean['nc2_equinormness_mean'],
            'nc2_equinormness_mean',
            linestyle='--',
            color=colors[0],
        )

    axes[0, 1].set_ylabel('nc2_equinormness_mean')
    ax011 = axes[0, 1].twinx()
    _plot_line(
        ax011,
        x,
        nc_val_mean['nc2_equiangularity_mean'],
        'nc2_equiangularity_mean',
        markers[1],
        color=colors[1],
        error=nc_val_std['nc2_equiangularity_mean'],
    )
    if nc_train_mean is not None:
        _plot_line(
            ax011,
            x,
            nc_train_mean['nc2_equiangularity_mean'],
            linestyle='--',
            color=colors[1],
        )
    ax011.set_ylabel('nc2_equiangularity_mean')
    # plot_line(axes[0, 1], x, nc['gnc2_hyperspherical_uniformity'], 'gnc2_hyperspherical_uniformity', markers[2])

    # Subplot 10 ======================================================================================
    _plot_line(
        axes[1, 0],
        x,
        nc_val_mean['nc3_self_duality'],
        'nc3_self_duality',
        markers[0],
        error=nc_val_std['nc3_self_duality'],
        color=colors[0],
    )
    if nc_train_mean is not None:
        _plot_line(
            axes[1, 0],
            x,
            nc_train_mean['nc3_self_duality'],
            linestyle='--',
            color=colors[0],
        )

    ax101 = axes[1, 0].twinx()
    _plot_line(
        ax101,
        x,
        nc_val_mean['unc3_uniform_duality_mean'],
        'unc3_uniform_duality_mean',
        markers[1],
        color=colors[1],
        error=nc_val_std['unc3_uniform_duality_mean'],
    )
    if nc_train_mean is not None:
        _plot_line(
            ax101,
            x,
            nc_train_mean['unc3_uniform_duality_mean'],
            linestyle='--',
            color=colors[1],
        )
    ax101.set_ylabel('unc3_mean')
    axes[1, 0].set_ylabel('nc3')

    # Subplot 11 ======================================================================================
    _plot_line(axes[1, 1], x, acc_train_mean, 'acc_train', 'None', error=acc_train_std)
    _plot_line(axes[1, 1], x, acc_val_mean, 'acc_val', 'None', error=acc_val_std)
    _plot_line(
        axes[1, 1],
        x,
        nc_val_mean['nc4_classifier_agreement'],
        'nc4_classifier_agreement',
        markers[0],
        error=nc_val_std['nc4_classifier_agreement'],
    )
    if nc_train_mean is not None:
        _plot_line(
            axes[1, 1],
            x,
            nc_train_mean['nc4_classifier_agreement'],
            linestyle='--',
            color=colors[2],
        )
    axes[1, 1].set_ylabel('agreement/accuracy')

    # Legend subplot 00 ======================================================================================
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
    lengths = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lengths), len(arrs)))
    arr.mask = True
    for idx, length in enumerate(arrs):
        arr[: len(length), idx] = np.squeeze(length)
    mu = arr.mean(axis=-1)
    std = arr.std(axis=-1)
    return mu, std


def plot_nc(benchmark_name):
    _, epochs, acc_val, acc_train, nc_val, _, _ = load_benchmark_data(
        benchmark_name, nc_split='val'
    )

    _, _, _, _, nc_train, _, _ = load_benchmark_data(benchmark_name, nc_split='train')

    if benchmark_name == 'imagenet':
        with_errorbars = False
        nc_val_mean = nc_val
        nc_val_std = nc_val
        acc_val_mean = acc_val
        acc_val_std = acc_val
        acc_train_mean = acc_train
        acc_train_std = acc_train

    else:
        # Split into runs
        idx = np.where(epochs == 1)[0][1:]
        epochs = np.split(epochs, indices_or_sections=idx)

        acc_val_splits = np.split(acc_val, indices_or_sections=idx)
        acc_val_mean, acc_val_std = tolerant_mean(acc_val_splits)
        acc_train_splits = np.split(acc_train, indices_or_sections=idx)
        acc_train_mean, acc_train_std = tolerant_mean(acc_train_splits)

        nc_val_split = {}
        for k, v in nc_val.items():
            nc_val_split[k] = np.split(v, indices_or_sections=idx)
        nc_train_split = {}
        for k, v in nc_train.items():
            nc_train_split[k] = np.split(v, indices_or_sections=idx)

        nc_val_mean = {}
        nc_val_std = {}
        for k, v in nc_val_split.items():
            nc_val_mean[k], nc_val_std[k] = tolerant_mean(v)

        nc_train_mean = {}
        nc_train_std = {}
        for k, v in nc_train_split.items():
            nc_train_mean[k], nc_train_std[k] = tolerant_mean(v)

        epochs, _ = tolerant_mean(epochs)

        with_errorbars = True

    save_dir = path.res_plots / 'nc'
    save_dir.mkdir(exist_ok=True, parents=True)

    fig = _plot_grid(
        nc_val_mean,
        nc_val_std,
        acc_val_mean,
        acc_val_std,
        acc_train_mean,
        acc_train_std,
        epochs,
        'epoch',
        nc_train_std=nc_train_std,
        nc_train_mean=nc_train_mean,
        with_errorbars=with_errorbars,
    )
    _save_plot(fig, save_dir, f'nc_{benchmark_name}_epoch')

    fig = _plot_grid(
        nc_val_mean,
        nc_val_std,
        acc_val_mean,
        acc_val_std,
        acc_train_mean,
        acc_train_std,
        acc_val_mean,
        'acc_val',
        nc_train_std=nc_train_std,
        nc_train_mean=nc_train_mean,
        with_errorbars=with_errorbars,
    )
    _save_plot(fig, save_dir, f'nc_{benchmark_name}_acc')


def plot_nc_ratios(benchmark_name):
    _, epochs, acc_val, acc_train, nc_val, _, _ = load_benchmark_data(
        benchmark_name, nc_split='val'
    )

    _, _, _, _, nc_train, _, _ = load_benchmark_data(benchmark_name, nc_split='train')

    if benchmark_name == 'imagenet':
        with_errorbars = False
        nc_val_mean = nc_val
        nc_val_std = nc_val
        acc_val_mean = acc_val
        acc_val_std = acc_val
        acc_train_mean = acc_train
        acc_train_std = acc_train

    else:
        # Split into runs
        idx = np.where(epochs == 1)[0][1:]
        epochs = np.split(epochs, indices_or_sections=idx)

        acc_val_splits = np.split(acc_val, indices_or_sections=idx)
        acc_val_mean, acc_val_std = tolerant_mean(acc_val_splits)
        acc_train_splits = np.split(acc_train, indices_or_sections=idx)
        acc_train_mean, acc_train_std = tolerant_mean(acc_train_splits)

        nc_val_split = {}
        for k, v in nc_val.items():
            nc_val_split[k] = np.split(v, indices_or_sections=idx)
        nc_train_split = {}
        for k, v in nc_train.items():
            nc_train_split[k] = np.split(v, indices_or_sections=idx)

        nc_val_mean = {}
        nc_val_std = {}
        for k, v in nc_val_split.items():
            nc_val_mean[k], nc_val_std[k] = tolerant_mean(v)

        nc_train_mean = {}
        nc_train_std = {}
        for k, v in nc_train_split.items():
            nc_train_mean[k], nc_train_std[k] = tolerant_mean(v)

        epochs, _ = tolerant_mean(epochs)

        with_errorbars = True

    ratio_mean = {}
    for k, v_train in nc_train_mean.items():
        v_val = nc_val_mean[k]
        ratio_mean[k] = v_train / v_val

    ratio_std = {}
    for k, v_train in nc_train_std.items():
        v_val = nc_val_std[k]
        ratio_std[k] = 1 / (1 / v_train + 1 / v_val)

    save_dir = path.res_plots / 'nc_ratio'
    save_dir.mkdir(exist_ok=True, parents=True)

    fig = _plot_grid(
        ratio_mean,
        ratio_std,
        acc_val_mean,
        acc_val_std,
        acc_train_mean,
        acc_train_std,
        epochs,
        'epoch',
        with_errorbars=with_errorbars,
    )
    _save_plot(fig, save_dir, f'nc_{benchmark_name}_epoch')

    fig = _plot_grid(
        ratio_mean,
        ratio_std,
        acc_val_mean,
        acc_val_std,
        acc_train_mean,
        acc_train_std,
        acc_val_mean,
        'acc_val',
        with_errorbars=with_errorbars,
    )
    _save_plot(fig, save_dir, f'nc_{benchmark_name}_acc')


if __name__ == '__main__':
    # plot_nc('cifar10')
    # plot_nc('cifar100')
    # plot_nc('imagenet200')
    # plot_nc('imagenet')
    # plot_nc('alexnet')
    # plot_nc('vgg')
    # plot_nc('mobilenet')

    # plot_nc_ratios('cifar10')
    # plot_nc_ratios('cifar100')
    # plot_nc_ratios('imagenet200')
    # plot_nc_ratios('imagenet')
    plot_nc_ratios('alexnet')
    plot_nc_ratios('vgg')
    plot_nc_ratios('mobilenet')
