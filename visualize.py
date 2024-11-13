from collections import defaultdict
import json

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import path
from summarize_json import summarize_json


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


def get_acc(benchmark_name, run_id, split='val', filter_epochs=None):
    json_dir = path.ckpt_root / benchmark_name / run_id
    with open(json_dir / 'data.json', 'r') as f:
        data = json.load(f)

    acc_values = np.array(data['metrics']['Accuracy'][split]['values'])
    acc_epochs = np.array(data['metrics']['Accuracy'][split]['epochs']) + 1

    if benchmark_name == 'cifar10' and run_id in ['run0', 'run1']:
        acc_epochs -= 1

    if filter_epochs is not None:
        filter_epochs = np.array(filter_epochs)
        return acc_values[np.isin(acc_epochs, filter_epochs)], filter_epochs
    else:
        return acc_values, acc_epochs


def get_acc_nc_ood(benchmark_name,
                   acc_split='val',
                   nc_metric='nc1_cdnv',
                   ood_metric='AUROC'):

    benchmark_dir = path.res_data / benchmark_name
    data = []
    run_ids = []

    for run_dir in benchmark_dir.glob('run*'):
        ckpt_dirs = natsorted(list(run_dir.glob('e*')), key=str)
        acc_val, acc_epoch = get_acc(benchmark_name, run_dir.name, acc_split)
        run_number = int(run_dir.name[3:])

        for ckpt_dir in ckpt_dirs:
            with pd.HDFStore(ckpt_dir / 'metrics.h5') as store:
                nc_df = store.get('nc')
                nc = nc_df.iloc[0][nc_metric]

                epoch = int(ckpt_dir.name[1:])
                acc = acc_val[acc_epoch == epoch][0]

                ood_keys = list(store.keys())
                ood_keys.remove('/nc')
                near_ood = []
                far_ood = []
                for k in ood_keys:
                    ood_df = store.get(k)
                    near_ood.append(ood_df.at['nearood', ood_metric])
                    far_ood.append(ood_df.at['farood', ood_metric])

                near_ood = np.mean(np.array(near_ood))
                far_ood = np.mean(np.array(far_ood))

                data.append([acc, nc, near_ood, far_ood])
                run_ids.append(run_number)

    return np.array(data), np.array(run_ids)


def plot_acc_nc_ood(benchmark_name,
                    acc_split='val',
                    nc_metric='nc1_cdnv',
                    ood_metric='AUROC'):

    data, run_ids = get_acc_nc_ood(benchmark_name,
                                   acc_split=acc_split,
                                   nc_metric=nc_metric,
                                   ood_metric=ood_metric)

    labels = ['acc', 'nc', 'nearood', 'farood']

    def plot_cut(x, y, z, resolution=100, cuts=3):
        kde = KDEMultivariate(data[:, [x, y, z]], 'ccc')

        x_lin = np.linspace(data[:, x].min(), data[:, x].max(), resolution)
        y_lin = np.linspace(data[:, y].min(), data[:, y].max(), resolution)
        z_lin = np.linspace(data[:, z].min(), data[:, z].max(), cuts**2)

        X, Y = np.meshgrid(x_lin, y_lin)
        ones = np.ones_like(X)

        res = []
        for z_ in z_lin:
            Z = z_ * ones
            points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            pdf = kde.pdf(points).reshape(X.shape)
            res.append(pdf)

        fig, axes = plt.subplots(cuts, cuts, figsize=(10, 10))

        for ax, pdf in zip(axes.ravel(), res):
            ax.contourf(X, Y, pdf)
            ax.axis('off')

        plt.tight_layout()
        fig.suptitle(f'{labels[z]}  {labels[x]}-{labels[y]}')
        
        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'acc_{acc_split}_{nc_metric}_{ood_metric}_{labels[z]}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_cut(0, 1, 2)
    plot_cut(0, 1, 3)

    def plot_corr(z, label):
        acc = data[:, 0]
        nc = data[:, 1]
        ood = data[:, z]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        c = [colors[i] for i in run_ids]

        axes.ravel()[0].scatter(acc, nc, c=c, marker='o')
        axes.ravel()[0].set_xlabel(f'acc {acc_split}')
        axes.ravel()[0].set_ylabel(nc_metric)

        axes.ravel()[1].scatter(acc, ood, c=c, marker='o')
        axes.ravel()[1].set_xlabel(f'acc {acc_split}')
        axes.ravel()[1].set_ylabel(f'{ood_metric} {label}')

        axes.ravel()[2].scatter(nc, ood, c=c, marker='o')
        axes.ravel()[2].set_xlabel(nc_metric)
        axes.ravel()[2].set_ylabel(f'{ood_metric} {label}')

        plt.tight_layout()

        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_{label}.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_corr(2, 'nearood')
    plot_corr(3, 'farood')

    def plot_corr_matrix():
        df_near = pd.DataFrame(data[:, [0, 1, 2]])
        df_far = pd.DataFrame(data[:, [0, 1, 3]])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im0 = axes[0].matshow(df_near.corr())
        axes[0].set_title('nearood')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].matshow(df_far.corr())
        axes[1].set_title('farood')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()

        save_path = path.res_plots / benchmark_name
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'corr_matrix.png'
        plt.savefig(save_path / filename, bbox_inches='tight')
        plt.close()

    plot_corr_matrix()


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
    
    try:
        for ood_key in near_ood.keys():
            plt.plot(nc, near_ood[ood_key], '-', alpha=0.3, color=colors[0])
            plt.plot(nc, far_ood[ood_key], '-', alpha=0.3, color=colors[1])
            plt.plot(nc, near_ood[ood_key], 'o', color=colors[0], label='nearood')
            plt.plot(nc, far_ood[ood_key], 'o', color=colors[1], label='farood')
    except ValueError as e:
        print(run_id)
        print(nc)
        print(ood_key)
        print(near_ood[ood_key])
        print(far_ood[ood_key])
        raise e

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


def plot_acc_ood(benchmark_name,
                 run_id,
                 acc_split='val',  # train or val
                 ood_metric='AUROC'):

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

    acc = []
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
                near_ood[k].append(ood_df.at['nearood', ood_metric])
                far_ood[k].append(ood_df.at['farood', ood_metric])

    acc, epoch = get_acc(benchmark_name, run_id, split=acc_split, filter_epochs=epoch)
    
    for ood_key in near_ood.keys():
        plt.plot(acc, near_ood[ood_key], '-', alpha=0.3, color=colors[0])
        plt.plot(acc, far_ood[ood_key], '-', alpha=0.3, color=colors[1])
        plt.plot(acc, near_ood[ood_key], 'o', color=colors[0], label='nearood')
        plt.plot(acc, far_ood[ood_key], 'o', color=colors[1], label='farood')

    plt.title(f'{benchmark_name} {run_id}')
    plt.xlabel(f'acc {acc_split}')
    plt.ylabel(ood_metric)
    
    # https://stackoverflow.com/a/13589144
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    save_path = path.res_plots / benchmark_name / run_id
    save_path.mkdir(exist_ok=True, parents=True)
    filename = f'acc_{acc_split}_{ood_metric}.png'
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()


def plot_nc(benchmark_name,
            run_id):

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

    epoch = []
    nc = defaultdict(list)

    for json_dir in ckpt_dirs:
        with pd.HDFStore(json_dir / 'metrics.h5') as store:
            epoch.append(int(json_dir.name[1:]))
            nc_df = store.get('nc')
            for name, value in nc_df.items():
                nc[name].append(value)
    epoch = np.array(epoch)
    
    def plot_line(ax, x, y, label, marker, color=None):
        if color is None:
            ax.plot(x, y, label=label, marker=marker, markersize=5)
        else:
            ax.plot(x, y, label=label, marker=marker, markersize=5, color=color)

    # acc_train_values_all, acc_train_epochs_all = get_acc(benchmark_name, run_id, split='train')
    acc_val_values_all, acc_val_epochs_all = get_acc(benchmark_name, run_id, split='val')
    acc_train_values_filtered, _ = get_acc(benchmark_name, run_id, split='val', filter_epochs=epoch)

    def plot_nc_(x, x_label):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Subplot 00
        plot_line(axes[0, 0], x, nc['nc1_weak_between'], 'nc1_weak_between', markers[1])
        plot_line(axes[0, 0], x, nc['nc1_weak_within'], 'nc1_weak_within', markers[2])
        plot_line(axes[0, 0], x, nc['nc1_cdnv'], 'nc1_cdnv', markers[3])
        ax001 = axes[0, 0].twinx()
        plot_line(ax001, x, nc['nc1_strong'], 'nc1_strong', markers[0], color=colors[3])
        ax001.set_ylabel('strong')
        axes[0, 0].set_ylabel('other')
        # Subplot 01
        plot_line(axes[0, 1], x, nc['nc2_equinormness'], 'nc2_equinormness', markers[0])
        plot_line(axes[0, 1], x, nc['nc2_equiangularity'], 'nc2_equiangularity', markers[1])
        plot_line(axes[0, 1], x, nc['gnc2_hyperspherical_uniformity'], 'gnc2_hyperspherical_uniformity', markers[2])
        # Subplot 10
        plot_line(axes[1, 0], x, nc['nc3_self_duality'], 'nc3_self_duality', markers[0])
        ax011 = axes[1, 0].twinx()
        plot_line(ax011, x, nc['unc3_uniform_duality'], 'unc3_uniform_duality', markers[1], color=colors[1])
        ax011.set_ylabel('unc3')
        axes[1, 0].set_ylabel('nc3')
        # Subplot 11
        plot_line(axes[1, 1], x, nc['nc4_classifier_agreement'], 'nc4_classifier_agreement', markers[0])
        ax111 = axes[1, 1].twinx()
        if x_label == 'epoch':
            #plot_line(ax111, acc_train_epochs_all, acc_train_values_all, 'acc train', 'None', color=colors[1])
            plot_line(ax111, acc_val_epochs_all, acc_val_values_all, 'acc val', 'None', color=colors[2])
            ax111.set_ylabel('accuracy')
        axes[1, 1].set_ylabel('agreement')

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
        if x_label == 'epoch':
            lines110, labels110 = axes[1, 1].get_legend_handles_labels()
            lines111, labels111 = ax111.get_legend_handles_labels()
            ax111.legend(lines110 + lines111, labels110 + labels111)
        else:
            axes[1, 1].legend()

        axes[0, 0].set_title('NC1')
        axes[0, 1].set_title('NC2')
        axes[1, 0].set_title('NC3')
        axes[1, 1].set_title('NC4')

        axes[0, 0].set_xlabel(x_label)
        axes[0, 1].set_xlabel(x_label)
        axes[1, 0].set_xlabel(x_label)
        axes[1, 1].set_xlabel(x_label)

        plt.tight_layout()

        save_path = path.res_plots / benchmark_name / run_id
        save_path.mkdir(exist_ok=True, parents=True)
        filename = f'nc_{x_label}.png'
        plt.savefig(save_path / filename)
        plt.close()

    plot_nc_(epoch, 'epoch')
    plot_nc_(acc_train_values_filtered, 'acc_train')


def plot_ood(benchmark_name,
             run_id,
             ood_metric='AUROC'):
    
    nc_ood_methods = ['nusa', 'vim', 'ncscore', 'neco', 'epa']

    main_dir = path.res_data / benchmark_name / run_id
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

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
    ckpt_dirs = natsorted(list(main_dir.glob('e*')), key=str)

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
    plot_acc_ood(benchmark_name, run_id, acc_split='val')
    # plot_acc_ood(benchmark_name, run_id, acc_split='train')
    plot_nc(benchmark_name, run_id)
    plot_ood(benchmark_name, run_id)
    plot_ood_combined(benchmark_name, run_id)
    plot_acc_nc_ood(benchmark_name)


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    
    # cfg.benchmark = 'cifar10'
    # cfg.run = 'run0'

    if 'run' in cfg:
        plot_all(cfg.benchmark, cfg.run)
    else:
        plot_acc_nc_ood(cfg.benchmark)
