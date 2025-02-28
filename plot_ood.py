import matplotlib.pyplot as plt
import numpy as np

import path
from plot_nc import tolerant_mean
from plot_utils import (
    colors,
    load_benchmark_data,
    metric_markers,
)


def _plot(
    benchmark_name, save_path, filename, ood_keys, data, x_axis, acc_split, ood_metric
):
    epochs, acc_mean, nood_mean, nood_std, food_mean, food_std = data

    # Plotting

    def sub_plot(ax, avg_ood, color):
        for ood_key in ood_keys:
            y_values = avg_ood[ood_key]

            zipped = (
                zip(epochs, y_values) if x_axis == 'epoch' else zip(acc_mean, y_values)
            )

            # Remove NaNs from data
            valid_data = [(x, y) for x, y in zipped if not np.isnan(y)]
            if not valid_data:
                continue
            x_values, y_values = zip(*valid_data)
            ax.plot(x_values, y_values, '-', alpha=0.3, color=color)
            ax.plot(
                x_values, y_values, metric_markers[ood_key], color=color, label=ood_key
            )

    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    sub_plot(axes[0], nood_mean, colors[0])
    sub_plot(axes[1], food_mean, colors[1])

    if x_axis == 'epoch':
        axes[0].set_xlabel('epoch')
        axes[0].set_xscale('log')
        axes[1].set_xlabel('epoch')
        axes[1].set_xscale('log')
    else:
        axes[0].set_xlabel(f'acc {acc_split}')
        axes[1].set_xlabel(f'acc {acc_split}')

    axes[0].set_ylabel(ood_metric)
    axes[1].set_ylabel(ood_metric)

    plt.suptitle(f'{benchmark_name} {"epoch" if x_axis == "epoch" else "accuracy"}')

    # Create legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=2)

    plt.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_ood(
    benchmark_name,
    nc_split='val',
    acc_split='val',
    ood_metric='AUROC',
):
    _, epochs, acc_val, _, _, nood, food, _ = load_benchmark_data(
        benchmark_name, nc_split=nc_split, ood_metric=ood_metric
    )

    # Split into runs
    idx = np.where(epochs == 1)[0][1:]

    epochs = np.split(epochs, indices_or_sections=idx)
    epochs, _ = tolerant_mean(epochs)

    for k, v in nood.items():
        nood[k] = np.split(v, indices_or_sections=idx)

    for k, v in food.items():
        food[k] = np.split(v, indices_or_sections=idx)

    if benchmark_name == 'imagenet':
        acc_mean = acc_val
    else:
        acc = np.split(acc_val, indices_or_sections=idx)
        acc_mean, _ = tolerant_mean(acc)

    nood_mean = {}
    nood_std = {}
    for k, v in nood.items():
        nood_mean[k], nood_std[k] = tolerant_mean(v)

    food_mean = {}
    food_std = {}
    for k, v in food.items():
        food_mean[k], food_std[k] = tolerant_mean(v)

    save_dir = path.res_plots / 'ood'
    save_dir.mkdir(parents=True, exist_ok=True)

    data = (epochs, acc_mean, nood_mean, nood_std, food_mean, food_std)

    _plot(
        benchmark_name,
        save_dir,
        f'ood_{benchmark_name}_acc',
        nood.keys(),
        data,
        None,
        acc_split,
        ood_metric,
    )
    _plot(
        benchmark_name,
        save_dir,
        f'ood_{benchmark_name}_epoch',
        nood.keys(),
        data,
        'epoch',
        acc_split,
        ood_metric,
    )


if __name__ == '__main__':
    plot_ood('cifar10')
    plot_ood('cifar100')
    plot_ood('imagenet200')
    plot_ood('imagenet')
    plot_ood('noise')
    plot_ood('alexnet')
    plot_ood('vgg')
    plot_ood('mobilenet')
