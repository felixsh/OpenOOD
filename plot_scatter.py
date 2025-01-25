import matplotlib.pyplot as plt
import numpy as np

import path
from plot_utils import nc_metrics, load_benchmark_data


def _plot(acc, nc, ood, color_id, nc_metric, ood_metric, ood_label):
    assert len(acc) == len(nc) == len(ood), f'{len(acc)} {len(nc)} {len(ood)}'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes.ravel()[0].scatter(acc, nc, c=color_id, marker='o')
    axes.ravel()[0].set_xlabel(f'acc val')
    axes.ravel()[0].set_ylabel(nc_metric)

    axes.ravel()[1].scatter(acc, ood, c=color_id, marker='o')
    axes.ravel()[1].set_xlabel(f'acc val')
    axes.ravel()[1].set_ylabel(f'{ood_metric} {ood_label}')

    axes.ravel()[2].scatter(nc, ood, c=color_id, marker='o')
    axes.ravel()[2].set_xlabel(nc_metric)
    axes.ravel()[2].set_ylabel(f'{ood_metric} {ood_label}')

    plt.tight_layout()
    return fig


def _save(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    fig.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def _mean(a1, a2):
    x = np.vstack((a1, a2))
    return x.mean(axis=0)


def plot_scatter_all(benchmark_name, nc_split='val', ood_metric='AUROC'):
    print(f'plotting scatter {benchmark_name} {ood_metric} ...')
    _, epochs, acc, nc_dict, nood_dict, food_dict, save_dir = load_benchmark_data(benchmark_name, nc_split, ood_metric)

    # Mean over ood methods
    nood_values = list(np.mean([v for v in nood_dict.values()], axis=0))
    food_values = list(np.mean([v for v in food_dict.values()], axis=0))

    # Epochs to color index
    _, color_id = np.unique(epochs, return_inverse=True)
    color_id += 1

    save_dir = path.res_plots / 'scatter' / benchmark_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for nc_metric in nc_metrics:
        print(nc_metric)
        nc_values = nc_dict[nc_metric]
        # Plot and save
        fig = _plot(acc, nc_values, _mean(nood_values, food_values), color_id, nc_metric, ood_metric, 'mean')
        _save(fig, save_dir, f'mean_{nc_metric}_{ood_metric}')
        fig = _plot(acc, nc_values, nood_values, color_id, nc_metric, ood_metric, 'near')
        _save(fig, save_dir, f'scatter_near_{nc_metric}_{ood_metric}')
        fig = _plot(acc, nc_values, food_values, color_id, nc_metric, ood_metric, 'far')
        _save(fig, save_dir, f'scatter_far_{nc_metric}_{ood_metric}')


if __name__ == '__main__':
    # plot_scatter_all('cifar10')
    # plot_scatter_all('cifar100')
    plot_scatter_all('imagenet200')
    # plot_scatter_all('imagenet')
    # plot_scatter_all('alexnet')
    # plot_scatter_all('mobilenet')
    # plot_scatter_all('vgg')