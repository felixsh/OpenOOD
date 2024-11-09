from collections import defaultdict
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import path


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

markers = [
    'o',  # Circle
    's',  # Square
    'D',  # Diamond
    '^',  # Upward triangle
    'v',  # Downward triangle
    '<',  # Left triangle
    '>',  # Right triangle
    'p',  # Pentagon
    '*',  # Star
    'h',  # Hexagon
    'X',  # X-shaped marker
    '+',  # Plus sign
    'x',  # X mark
    '|',  # Vertical line
    '_',  # Horizontal line
]


def plot_nc_ood(benchmark_name,
                run_id,
                nc_metric='nc1_cdnv',
                ood_metric='AUROC'):

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = sorted(list(main_dir.glob('e*')))

    nc = []
    epoch = []
    near_ood = defaultdict(list)
    far_ood = defaultdict(list)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            epoch.append(int(re.search(r'e(\d+)$', ckpt_dir.name).group(1)))
            nc_df = store.get('nc')
            nc.append(nc_df.iloc[0][nc_metric])

            ood_keys = list(store.keys())
            ood_keys.remove('/nc')
            for k in ood_keys:
                ood_df = store.get(k)
                near_ood[k].append(ood_df.at['nearood', ood_metric])
                far_ood[k].append(ood_df.at['farood', ood_metric])

    epoch = np.array(epoch)
    
    for ood_key in near_ood.keys():
        plt.plot(nc, near_ood[ood_key], '-', alpha=0.3, color=colors[0])
        plt.plot(nc, far_ood[ood_key], '-', alpha=0.3, color=colors[1])
        plt.plot(nc, near_ood[ood_key], 'o', color=colors[0], label='nearood')
        plt.plot(nc, far_ood[ood_key], 'o', color=colors[1], label='farood')

    plt.title(f'{benchmark_name} {run_id}')
    plt.xlabel(nc_metric)
    plt.ylabel(ood_metric)
    
    # https://stackoverflow.com/a/13589144
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    save_path = path.res_plots / benchmark_name / run_id
    save_path.mkdir(exist_ok=True, parents=True)
    filename = f'{nc_metric}_{ood_metric}.png'
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()


def plot_nc(benchmark_name,
            run_id):

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = sorted(list(main_dir.glob('e*')))

    epoch = []
    nc = defaultdict(list)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            epoch.append(int(re.search(r'e(\d+)$', ckpt_dir.name).group(1)))
            nc_df = store.get('nc')
            for name, value in nc_df.items():
                nc[name].append(value)
    
    def plot_line(ax, values, label, marker):
        ax.plot(values, label=label, marker=marker, markersize=5)
    

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plot_line(axes[0, 0], nc['nc1_strong'], 'nc1_strong', markers[0])
    plot_line(axes[0, 0], nc['nc1_weak_between'], 'nc1_weak_between', markers[1])
    plot_line(axes[0, 0], nc['nc1_weak_within'], 'nc1_weak_within', markers[2])
    plot_line(axes[0, 0], nc['nc1_cdnv'], 'nc1_cdnv', markers[3])

    plot_line(axes[0, 1], nc['nc2_equinormness'], 'nc2_equinormness', markers[0])
    plot_line(axes[0, 1], nc['nc2_equiangularity'], 'nc2_equiangularity', markers[1])
    plot_line(axes[0, 1], nc['gnc2_hyperspherical_uniformity'], 'gnc2_hyperspherical_uniformity', markers[2])

    plot_line(axes[1, 0], nc['nc3_self_duality'], 'nc3_self_duality', markers[0])
    plot_line(axes[1, 0], nc['unc3_uniform_duality'], 'unc3_uniform_duality', markers[1])

    plot_line(axes[1, 1], nc['nc4_classifier_agreement'], 'nc4_classifier_agreement', markers[0])
        
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()

    axes[0, 0].set_title('NC1')
    axes[0, 1].set_title('NC2')
    axes[1, 0].set_title('NC3')
    axes[1, 1].set_title('NC4')
    
    axes[0, 0].set_xlabel('epoch')
    axes[0, 1].set_xlabel('epoch')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 1].set_xlabel('epoch')

    plt.tight_layout()

    save_path = path.res_plots / benchmark_name / run_id
    save_path.mkdir(exist_ok=True, parents=True)
    filename = 'nc_all.png'
    plt.savefig(save_path / filename)
    plt.close()


if __name__ == '__main__':
    # plot_nc_ood('cifar10', 'run0')
    plot_nc('cifar10', 'run0')
