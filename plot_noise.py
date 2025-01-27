import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import path
from plot_utils import load_noise_data, mean_ood_values, nc_metrics


def plot_noise(nc_split='val', ood_metric='AUROC'):
    noise_lvl, _, acc, nc_dict, nood_dict, food_dict, _ = load_noise_data(nc_split, ood_metric)

    data = {
        'noise level': noise_lvl,
        'accuracy': acc,
        f'{ood_metric}': mean_ood_values(nood_dict, food_dict),
    }
    data |= nc_dict
    data = {k: v.ravel() for k, v in data.items()}
    df = pd.DataFrame(data)

    _, axes = plt.subplots(3, 4, figsize=(12, 9))
    ax = axes.ravel()

    sns.scatterplot(ax=ax[0], data=df, x='noise level', y='accuracy')
    sns.scatterplot(ax=ax[1], data=df, x='noise level', y=ood_metric)
    for a, nc_metric in zip(ax[2:], nc_metrics):
        sns.scatterplot(ax=a, data=df, x='noise level', y=nc_metric)

    plt.tight_layout()

    save_dir = path.res_plots / 'noise'
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f'noise_{nc_split}_{ood_metric}'
    plt.savefig(save_dir / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_noise()
