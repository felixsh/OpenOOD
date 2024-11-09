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

nc_metrics = [
    'nc1_strong',
    'nc1_weak_between',
    'nc1_weak_within',
    'nc1_cdnv',
    'nc2_equinormness',
    'nc2_equiangularity',
    'gnc2_hyperspherical_uniformity',
    'nc3_self_duality',
    'unc3_uniform_duality',
    'nc4_classifier_agreement'
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


if __name__ == '__main__':
    plot_nc_ood('cifar10', 'run0')
