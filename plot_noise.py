from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import path
from plot_utils import load_acc, load_noise, load_nc, load_ood


def load_nc_last(run_dirs, nc_metric):
    data = []
    for r in run_dirs:
        nc, _ = load_nc(r)
        data.append(nc[nc_metric][-1])
    return np.array(data)


def load_acc_last(run_dirs):
    data = []
    for r in run_dirs:
        acc = load_acc(r)
        data.append(acc['val']['values'][-1])
    return np.array(data)


def load_ood_last(run_dirs, ood_metric):
    near = []
    far = []
    for r in run_dirs:
        n, f, _ = load_ood(r, ood_metric)
        near.append(np.mean([v[-1] for v in n.values()]))
        far.append(np.mean([v[-1] for v in f.values()]))
    return np.array(near), np.array(far)


def plot_noise_():
    main_dir = Path('/mrtstorage/users/hauser/openood_res/data/cifar10/NCResNet18_32x32/noise/300+_epochs')
    nc_metric='nc1_cdnv'
    ood_metric='AUROC'

    run_dirs = [p for p in main_dir.iterdir() if p.is_dir()]

    noise_lvl =  np.array([load_noise(r) for r in run_dirs])
    nc = load_nc_last(run_dirs, nc_metric)
    acc = load_acc_last(run_dirs)
    nearood, farood = load_ood_last(run_dirs, ood_metric)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes.ravel()[0].scatter(noise_lvl, acc, marker='o')
    axes.ravel()[0].set_xlabel('noise level')
    axes.ravel()[0].set_ylabel('accuracy')

    axes.ravel()[1].scatter(noise_lvl, nc, marker='o')
    axes.ravel()[1].set_xlabel('noise level')
    axes.ravel()[1].set_ylabel(f'{nc_metric}')

    axes.ravel()[2].scatter(noise_lvl, nearood, marker='o')
    axes.ravel()[2].set_xlabel('noise level')
    axes.ravel()[2].set_ylabel(f'{ood_metric} nearood')

    axes.ravel()[3].scatter(noise_lvl, farood, marker='o')
    axes.ravel()[3].set_xlabel('noise level')
    axes.ravel()[3].set_ylabel(f'{ood_metric} farood')

    plt.tight_layout()

    save_dir = path.res_plots / main_dir.relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f'noise_{nc_metric}_{ood_metric}'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_noise():
    main_dir = Path('/mrtstorage/users/hauser/openood_res/data/cifar10/NCResNet18_32x32/noise/300+_epochs')
    nc_metric='nc1_cdnv'
    ood_metric='AUROC'

    run_dirs = [p for p in main_dir.iterdir() if p.is_dir()]

    noise_lvl =  np.array([load_noise(r) for r in run_dirs])
    nc = load_nc_last(run_dirs, nc_metric)
    acc = load_acc_last(run_dirs)
    nearood, farood = load_ood_last(run_dirs, ood_metric)

    data = {
        'noise level': noise_lvl,
        nc_metric: nc,
        'accuracy': acc,
        f'{ood_metric} near': nearood,
        f'{ood_metric} far': farood,
    }
    data = {k: v.ravel() for k, v in data.items()}
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    ax = axes.ravel()

    sns.regplot(ax=ax[0], data=df, x='noise level', y='accuracy')
    sns.regplot(ax=ax[1], data=df, x='noise level', y=nc_metric)
    sns.regplot(ax=ax[2], data=df, x='noise level', y=f'{ood_metric} near')
    sns.regplot(ax=ax[3], data=df, x='noise level', y=f'{ood_metric} far')

    plt.tight_layout()

    save_dir = path.res_plots / main_dir.relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f'noise_{nc_metric}_{ood_metric}'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_noise()
