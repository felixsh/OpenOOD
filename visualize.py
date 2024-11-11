from collections import defaultdict
import re

import matplotlib.pyplot as plt
from natsort import natsorted, realsorted
import numpy as np
from omegaconf import OmegaConf
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
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

    nc = []
    epoch = []
    near_ood = defaultdict(list)
    far_ood = defaultdict(list)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            epoch.append(int(ckpt_dir.name[1:]))
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
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

    epoch = []
    nc = defaultdict(list)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            epoch.append(int(ckpt_dir.name[1:]))
            nc_df = store.get('nc')
            for name, value in nc_df.items():
                nc[name].append(value)
    
    def plot_line(ax, x, y, label, marker, color=None):
        if color is None:
            ax.plot(x, y, label=label, marker=marker, markersize=5)
        else:
            ax.plot(x, y, label=label, marker=marker, markersize=5, color=color)
    

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Subplot 00
    plot_line(axes[0, 0], epoch, nc['nc1_weak_between'], 'nc1_weak_between', markers[1])
    plot_line(axes[0, 0], epoch, nc['nc1_weak_within'], 'nc1_weak_within', markers[2])
    plot_line(axes[0, 0], epoch, nc['nc1_cdnv'], 'nc1_cdnv', markers[3])
    ax001 = axes[0, 0].twinx()
    plot_line(ax001, epoch, nc['nc1_strong'], 'nc1_strong', markers[0], color=colors[3])
    ax001.set_ylabel('strong')
    axes[0, 0].set_ylabel('other')
    # Subplot 01
    plot_line(axes[0, 1], epoch, nc['nc2_equinormness'], 'nc2_equinormness', markers[0])
    plot_line(axes[0, 1], epoch, nc['nc2_equiangularity'], 'nc2_equiangularity', markers[1])
    plot_line(axes[0, 1], epoch, nc['gnc2_hyperspherical_uniformity'], 'gnc2_hyperspherical_uniformity', markers[2])
    # Subplot 10
    plot_line(axes[1, 0], epoch, nc['nc3_self_duality'], 'nc3_self_duality', markers[0])
    ax011 = axes[1, 0].twinx()
    plot_line(ax011, epoch, nc['unc3_uniform_duality'], 'unc3_uniform_duality', markers[1], color=colors[1])
    ax011.set_ylabel('unc3')
    axes[1, 0].set_ylabel('nc3')
    # Subplot 11
    plot_line(axes[1, 1], epoch, nc['nc4_classifier_agreement'], 'nc4_classifier_agreement', markers[0])

    # Legend subplot 00
    lines000, labels000 = axes[0, 0].get_legend_handles_labels()
    lines001, labels001 = ax001.get_legend_handles_labels()
    ax001.legend(lines000 + lines001, labels000 + labels001)
    # Legend subplot 01
    axes[0, 1].legend()
    # Legend subplot 10
    lines100, labels100 = axes[1, 0].get_legend_handles_labels()
    lines101, labels101 = ax011.get_legend_handles_labels()
    ax011.legend(lines100 + lines101, labels100 + labels101)
    # Legend subplot 11
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


def plot_ood(benchmark_name,
             run_id,
             ood_metric='AUROC'):
    
    nc_ood_methods = ['nusa', 'vim', 'ncscore', 'neco', 'epa']

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')))

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            ood_keys = list(store.keys())
            ood_keys.remove('/nc')
            for k in ood_keys:
                color = colors[1] if k[1:] in nc_ood_methods else colors[0]
                label = 'nc method' if k[1:] in nc_ood_methods else 'baseline method'

                ood_df = store.get(k)
                near_ood = ood_df.at['nearood', ood_metric]
                far_ood = ood_df.at['farood', ood_metric]
                plt.plot(near_ood, far_ood, 'o', color=color, label=label)
        
        ax = plt.gca()
        x0 = ax.get_xlim()[0]
        y0 = ax.get_ylim()[0]
        ax.axline((x0, y0), slope=1, ls='--', color='k', alpha=0.5, zorder=1)

        plt.title(f'{benchmark_name} {run_id} {ckpt_dir.name} {ood_metric}')
        plt.xlabel('nearood')
        plt.ylabel('farood')
        
        # https://stackoverflow.com/a/13589144
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        ax.set_aspect('equal')

        save_path = path.res_plots / benchmark_name / run_id
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'{ood_metric}_{ckpt_dir.name}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()



def plot_ood_combined(benchmark_name,
                      run_id,
                      ood_metric='AUROC'):
    
    nc_ood_methods = ['nusa', 'vim', 'ncscore', 'neco', 'epa']

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')))

    epoch = []
    near_ood = defaultdict(list)
    far_ood = defaultdict(list)

    for ckpt_dir in ckpt_dirs:
        with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
            epoch.append(int(ckpt_dir.name[1:]))
            ood_keys = list(store.keys())
            ood_keys.remove('/nc')
            for k in ood_keys:
                ood_df = store.get(k)
                key = k[1:]
                near_ood[key].append(ood_df.at['nearood', ood_metric])
                far_ood[key].append(ood_df.at['farood', ood_metric])

    for k in near_ood.keys():
        color = colors[1] if k in nc_ood_methods else colors[0]
        label = 'nc method' if k in nc_ood_methods else 'baseline method'
        plt.plot(near_ood[k], far_ood[k], '-', color=color, alpha=0.3)
        plt.plot(near_ood[k], far_ood[k], 'o', color=color, label=label)

    ax = plt.gca()
    x0 = ax.get_xlim()[0]
    y0 = ax.get_ylim()[0]
    ax.axline((x0, y0), slope=1, ls='--', color='k', alpha=0.5, zorder=1)

    plt.title(f'{benchmark_name} {run_id} {ckpt_dir.name} {ood_metric}')
    plt.xlabel('nearood')
    plt.ylabel('farood')
        
    # https://stackoverflow.com/a/13589144
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax.set_aspect('equal')

    save_path = path.res_plots / benchmark_name / run_id
    save_path.mkdir(exist_ok=True, parents=True)
    filename = f'{ood_metric}_near_far.png'
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()


def plot_all(benchmark_name, run_id):
    plot_nc_ood(benchmark_name, run_id)
    plot_nc(benchmark_name, run_id)
    plot_ood(benchmark_name, run_id)
    plot_ood_combined(benchmark_name, run_id)


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    
    # cfg.benchmark = 'cifar10'
    # cfg.run = 'run0'

    plot_all(cfg.benchmark, cfg.run)
