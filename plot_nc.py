from pathlib import Path

import matplotlib.pyplot as plt

import path
from plot_utils import colors, markers
from plot_utils import load_acc, load_nc


def _plot_line(ax, x, y, label, marker, color=None):
    if color is None:
        ax.plot(x, y, label=label, marker=marker, markersize=5)
    else:
        ax.plot(x, y, label=label, marker=marker, markersize=5, color=color)


def _save_plot(fig, run_data_dir, filename):
    save_path = path.res_plots / run_data_dir.relative_to(path.res_data)
    save_path.mkdir(exist_ok=True, parents=True)

    fig.savefig(save_path / f'{filename}.png')
    fig.savefig(save_path / f'{filename}.pdf')
    plt.close()


def _plot_grid(nc, acc, x, x_label):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Subplot 00
    _plot_line(axes[0, 0], x, nc['nc1_weak_between'], 'nc1_weak_between', markers[1])
    _plot_line(axes[0, 0], x, nc['nc1_weak_within'], 'nc1_weak_within', markers[2])
    axes[0, 0].set_ylabel('nc1_weak')
    ax001 = axes[0, 0].twinx()
    _plot_line(ax001, x, nc['nc1_cdnv'], 'nc1_cdnv', markers[3], color=colors[2])
    ax001.set_ylabel('nc1_cdnv')
    #plot_line(ax001, x, nc['nc1_strong'], 'nc1_strong', markers[0], color=colors[3])

    # Subplot 01
    _plot_line(axes[0, 1], x, nc['nc2_equinormness'], 'nc2_equinormness', markers[0])
    axes[0, 1].set_ylabel('nc2_equinormness')
    ax011 = axes[0, 1].twinx()
    _plot_line(ax011, x, nc['nc2_equiangularity'], 'nc2_equiangularity', markers[1], color=colors[1])
    ax011.set_ylabel('nc2_equiangularity')
    # plot_line(axes[0, 1], x, nc['gnc2_hyperspherical_uniformity'], 'gnc2_hyperspherical_uniformity', markers[2])

    # Subplot 10
    _plot_line(axes[1, 0], x, nc['nc3_self_duality'], 'nc3_self_duality', markers[0])
    ax101 = axes[1, 0].twinx()
    _plot_line(ax101, x, nc['unc3_uniform_duality'], 'unc3_uniform_duality', markers[1], color=colors[1])
    ax101.set_ylabel('unc3')
    axes[1, 0].set_ylabel('nc3')

    # Subplot 11
    _plot_line(axes[1, 1], x, nc['nc4_classifier_agreement'], 'nc4_classifier_agreement', markers[0])
    if x_label == 'epoch':
        for k in acc.keys():
            _plot_line(axes[1, 1], acc[k]['epochs'], acc[k]['values'], f'acc {k}', 'None')
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


def plot_nc(run_data_dir):
    nc, epoch = load_nc(run_data_dir)
    acc = load_acc(run_data_dir)
    acc_filt = load_acc(run_data_dir, filter_epochs=epoch)

    # Limit to 500 for cifar100
    # epoch = epoch[:-1]
    # nc = {k: v[:-1] for k, v in nc.items()}

    fig = _plot_grid(nc, acc, epoch, 'epoch')
    _save_plot(fig, run_data_dir, 'nc_epoch')

    if 'train' in acc_filt:
        fig = _plot_grid(nc, acc, acc_filt['train']['values'], 'acc_train')
        _save_plot(fig, run_data_dir, 'nc_acc_train')

    if 'val' in acc_filt:
        fig = _plot_grid(nc, acc, acc_filt['val']['values'], 'acc_val')
        _save_plot(fig, run_data_dir, 'nc_acc_val')


def plot_nc_walk():
    # main_dir = Path('/mrtstorage/users/hauser/openood_res/data/cifar10/ResNet18_32x32/no_noise/300+_epochs')
    main_dir = Path('/mrtstorage/users/hauser/openood_res/data/cifar100/ResNet18_32x32/no_noise/1000+_epochs')
    run_dirs = [p for p in main_dir.iterdir() if p.is_dir()]
    for p in run_dirs:
        plot_nc(p)


if __name__ == '__main__':
    # run_dir = Path('/mrtstorage/users/hauser/openood_res/data/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07')
    # plot_nc(run_dir)

    plot_nc_walk()
