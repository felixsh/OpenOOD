from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from natsort import natsorted

import path
from eval_main import FeatureCache, get_run_ckpts
from plot_utils import (
    benchmark2ckptdirs,
    benchmark2loaddirs,
    load_noise,
    load_noise_data,
    mean_ood_1dict,
    mean_ood_2dict,
    nc_metrics_cov,
    nc_metrics_mean,
    ood_methods,
)


def plot_noise(nc_split='val', ood_metric='AUROC', reduction='mean'):
    noise_lvl, _, acc_val, acc_train, nc_dict, nood_dict, food_dict, _ = (
        load_noise_data(nc_split, ood_metric)
    )

    if reduction == 'mean':
        nc_metrics = nc_metrics_mean
    elif reduction == 'cov':
        nc_metrics = nc_metrics_cov
    else:
        raise NotImplementedError

    data = {
        'noise level': noise_lvl,
        'accuracy val': acc_val,
        'accuracy train': acc_train,
        ood_metric: mean_ood_2dict(nood_dict, food_dict),
    }
    data |= nc_dict
    data = {k: v.ravel() for k, v in data.items()}

    # data['unc3_uniform_duality'] = data.pop('unc3_uniform_duality_mean')

    df = pd.DataFrame(data)

    palette = sns.color_palette()

    _, axes = plt.subplots(3, 4, figsize=(12, 8))
    ax = axes.ravel()

    # for a in ax:
    #     a.set(xscale='log')
    #     a.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    sns.scatterplot(
        ax=ax[0], data=df, x='noise level', y='accuracy val', color=palette[2]
    )
    sns.scatterplot(
        ax=ax[1], data=df, x='noise level', y='accuracy train', color=palette[2]
    )
    sns.scatterplot(ax=ax[2], data=df, x='noise level', y=ood_metric, color=palette[1])
    for a, nc_metric in zip(ax[3:], nc_metrics):
        sns.scatterplot(ax=a, data=df, x='noise level', y=nc_metric, color=palette[0])

    # sns.scatterplot(ax=ax[0], data=df, x='noise level', y='accuracy val', color=palette[2])
    # sns.scatterplot(ax=ax[1], data=df, x='noise level', y=ood_metric, color=palette[1])
    # sns.scatterplot(ax=ax[2], data=df, x='noise level', y='nc1_weak_between', color=palette[0])
    # sns.scatterplot(ax=ax[3], data=df, x='noise level', y='unc3_uniform_duality', color=palette[0])

    plt.tight_layout()

    save_dir = path.res_plots / 'noise'
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f'noise_{nc_split}_{ood_metric}_{reduction}'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_noise_single(ood_metric='AUROC'):
    noise_lvl, _, acc_val, acc_train, _, nood_dict, food_dict, _ = load_noise_data(
        ood_metric=ood_metric
    )
    alpha = 0.5

    _, axes = plt.subplots(3, 4, figsize=(15, 10))
    ax = axes.ravel()

    # for a in ax:
    #     a.set(xscale='log')
    #     a.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax[0].scatter(noise_lvl, acc_train, color='tab:red', label='train', alpha=alpha)
    ax[0].scatter(noise_lvl, acc_val, color='tab:green', label='val', alpha=alpha)
    # ax[0].set_xlabel('noise level')
    # ax[0].set_ylabel('accuracy')
    ax[0].set_title('accuracy')
    ax[0].legend()

    for a, ood_method in zip(ax[1:], ood_methods):
        ood_method = ood_method[1:]
        a.scatter(
            noise_lvl, food_dict[ood_method], color='tab:blue', label='far', alpha=alpha
        )
        a.scatter(
            noise_lvl,
            nood_dict[ood_method],
            color='tab:orange',
            label='near',
            alpha=alpha,
        )
        # a.set_xlabel('noise level')
        # a.set_ylabel('AUROC')
        a.set_title(ood_method)

    ax[-1].legend()
    plt.tight_layout()

    save_dir = path.res_plots / 'noise_ood_methods'
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f'noise_single_{ood_metric}'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_noise_acc_ood(ood_metric='AUROC'):
    noise_lvl, _, acc_val, acc_train, _, nood_dict, food_dict, _ = load_noise_data(
        ood_metric=ood_metric
    )

    nearood = mean_ood_1dict(nood_dict)
    farood = mean_ood_1dict(food_dict)

    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes.ravel()

    ax[0].scatter(noise_lvl, acc_train, label='train')
    ax[0].scatter(noise_lvl, acc_val, label='val')
    ax[1].scatter(noise_lvl, farood, label='farood')
    ax[1].scatter(noise_lvl, nearood, label='nearood')

    ax[0].set_xlabel('noise level')
    ax[0].set_ylabel('accuracy')
    ax[0].legend()
    ax[1].set_xlabel('noise level')
    ax[1].set_ylabel(ood_metric)
    ax[1].legend()

    plt.tight_layout()

    save_dir = path.res_plots / 'noise'
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f'noise_acc_{ood_metric}'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_noise_feature_norm():
    main_dirs_load = benchmark2loaddirs['noise']
    main_dirs_load = [Path(p) for p in main_dirs_load]
    run_dirs_load = natsorted(
        [
            subdir
            for p in main_dirs_load
            if p.is_dir()
            for subdir in p.iterdir()
            if subdir.is_dir()
        ],
        key=str,
    )
    main_dirs_ckpt = benchmark2ckptdirs['noise']
    main_dirs_ckpt = [Path(p) for p in main_dirs_ckpt]
    run_dirs_ckpt = natsorted(
        [
            subdir
            for p in main_dirs_ckpt
            if p.is_dir()
            for subdir in p.iterdir()
            if subdir.is_dir()
        ],
        key=str,
    )
    print('number of runs', len(run_dirs_load), len(run_dirs_ckpt))
    assert len(run_dirs_load) == len(run_dirs_ckpt)

    res = []
    for r_load, r_ckpt in zip(run_dirs_load, run_dirs_ckpt):
        noise_lvl = load_noise(r_load)

        ckpt_path = get_run_ckpts(r_ckpt)[-1]
        feature_cache = FeatureCache('noise', ckpt_path)

        mean_feature_norm = feature_cache.mean_feature_norm()
        mean_feature_norm_g = feature_cache.mean_feature_norm(centered=True)
        mean_cluster_size = feature_cache.mean_cluster_size()
        mean_cluster_dist = feature_cache.mean_cluster_dist()
        mu_g_norm = feature_cache.mu_g_norm()

        res.append(
            [
                noise_lvl,
                mean_feature_norm,
                mean_feature_norm_g,
                mean_cluster_size,
                mean_cluster_dist,
                mu_g_norm,
            ]
        )

    (
        noise_lvl,
        mean_feature_norm,
        mean_feature_norm_g,
        mean_cluster_size,
        mean_cluster_dist,
        mu_g_norm,
    ) = zip(*res)

    plt.plot(noise_lvl, mean_feature_norm, 'o', label='feat')
    plt.plot(noise_lvl, mean_cluster_size, 'o', label='cluster size')
    plt.plot(noise_lvl, mean_feature_norm_g, 'o', label='feat center')
    plt.plot(noise_lvl, mean_cluster_dist, 'o', label='cluster dist')
    plt.plot(noise_lvl, mu_g_norm, 'o', label='mu_g')
    plt.xlabel('noise level')
    plt.ylabel('mean norm')
    plt.legend()

    plt.tight_layout()

    save_dir = path.res_plots / 'noise'
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = 'noise_feature_norm'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # plot_noise(reduction='cov')
    # plot_noise(reduction='mean')
    # plot_noise_acc_ood()
    # plot_noise_single()
    plot_noise_feature_norm()
