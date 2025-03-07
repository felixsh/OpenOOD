import matplotlib.pyplot as plt
import numpy as np

import path
from plot_nc import tolerant_mean
from plot_utils import (
    colors,
    load_benchmark_data,
    metric_markers,
)

benchmark2oodtasks = {
    'cifar10': ['cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100': ['cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o'],
    'imagenet': ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o'],
    'alexnet': ['cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'mobilenet': ['cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'vgg': ['cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
}


def _plot(
    benchmark_name,
    save_path,
    filename,
    ood_keys,
    data,
    x_axis,
    acc_split,
    ood_metric,
    acc_val=None,
    acc_train=None,
):
    epochs, acc_mean, nood_mean, nood_std, food_mean, food_std = data

    # Plotting

    def sub_plot(ax, avg_ood, color):
        for ood_key in ood_keys:
            if ood_key in ['knn', 'mds', 'neco', 'nusa']:
                continue

            y_values = avg_ood[ood_key]

            zipped = (
                zip(epochs, y_values) if x_axis == 'epoch' else zip(acc_mean, y_values)
            )

            alpha = 0.2 if ood_key in ['knn', 'mds', 'neco', 'nusa'] else 1.0

            # Remove NaNs from data
            valid_data = [(x, y) for x, y in zipped if not np.isnan(y)]
            if not valid_data:
                continue
            x_values, y_values = zip(*valid_data)
            ax.plot(x_values, y_values, '-', alpha=0.3, color=color)
            ax.plot(
                x_values,
                y_values,
                metric_markers[ood_key],
                color=color,
                label=ood_key,
                alpha=alpha,
            )

    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    sub_plot(axes[0], nood_mean, colors[0])
    sub_plot(axes[1], food_mean, colors[1])

    if x_axis == 'epoch':
        if acc_train is not None and acc_val is not None:
            ax0_twin = axes[0].twinx()
            ax1_twin = axes[1].twinx()
            ax0_twin.set_ylabel('accuracies')
            ax1_twin.set_ylabel('accuracies')

        if acc_train is not None:
            ax0_twin.plot(epochs, acc_train, '-', color=colors[2], label='acc train')
            ax1_twin.plot(epochs, acc_train, '-', color=colors[2], label='acc train')

        if acc_val is not None:
            ax0_twin.plot(epochs, acc_val, '-', color=colors[3], label='acc val')
            ax1_twin.plot(epochs, acc_val, '-', color=colors[3], label='acc val')

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

    filename = filename.replace('@', 'at')
    plt.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_ood(
    benchmark_name,
    nc_split='val',
    acc_split='val',
    ood_metric='AUROC',
):
    _, epochs, acc_val, acc_train, _, nood, food = load_benchmark_data(
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
        acc_val_mean = acc_val
        acc_train_mean = acc_train
    else:
        acc_val_split = np.split(acc_val, indices_or_sections=idx)
        acc_val_mean, _ = tolerant_mean(acc_val_split)
        acc_train_split = np.split(acc_train, indices_or_sections=idx)
        acc_train_mean, _ = tolerant_mean(acc_train_split)

    if acc_split == 'val':
        acc_mean = acc_val_mean
    elif acc_split == 'train':
        acc_mean = acc_train_mean

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
        f'ood_{benchmark_name}_{ood_metric}_acc',
        nood.keys(),
        data,
        None,
        acc_split,
        ood_metric,
        acc_train=acc_train_mean,
        acc_val=acc_val_mean,
    )
    _plot(
        benchmark_name,
        save_dir,
        f'ood_{benchmark_name}_{ood_metric}_epoch',
        nood.keys(),
        data,
        'epoch',
        acc_split,
        ood_metric,
        acc_train=acc_train_mean,
        acc_val=acc_val_mean,
    )


def mean_over_inner_entries(nested_dict):
    res = {}

    for outer_key, inner_dict in nested_dict.items():
        stacked_arrays = np.stack(
            [nested_dict[outer_key][key] for key in inner_dict.keys()]
        )
        res[outer_key] = np.mean(stacked_arrays, axis=0)

    return res


def _plot_specific(
    benchmark_name,
    save_path,
    filename,
    ood_tasks,
    data,
    x_axis,
    acc_split,
    ood_metric,
    acc_val=None,
    acc_train=None,
):
    epochs, acc_mean, ood_mean, ood_std = data
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = list(metric_markers.values())

    _ = plt.figure(figsize=(7, 5))
    for ood_task, color, marker in zip(ood_tasks, colors, markers):
        y_values = ood_mean[ood_task]

        zipped = zip(epochs, y_values) if x_axis == 'epoch' else zip(acc_mean, y_values)

        # Remove NaNs from data
        valid_data = [(x, y) for x, y in zipped if not np.isnan(y)]
        if not valid_data:
            continue
        x_values, y_values = zip(*valid_data)
        plt.plot(x_values, y_values, '-', alpha=0.3)
        plt.plot(
            x_values,
            y_values,
            marker,
            label=ood_task,
            color=color,
        )
    lines, labels = plt.gca().get_legend_handles_labels()
    plt.ylabel(ood_metric)
    plt.xlabel('epoch')
    plt.gca().set_xscale('log')

    if x_axis == 'epoch':
        if acc_train is not None and acc_val is not None:
            ax0_twin = plt.gca().twinx()
            ax0_twin.set_ylabel('accuracies')

        if acc_train is not None:
            ax0_twin.plot(epochs, acc_train, '-', color=colors[2], label='acc train')

        if acc_val is not None:
            ax0_twin.plot(epochs, acc_val, '-', color=colors[3], label='acc val')

        lines2, labels2 = ax0_twin.get_legend_handles_labels()
        lines.extend(lines2)
        labels.extend(labels2)
    else:
        plt.gca().set_xlabel(f'acc {acc_split}')

    plt.suptitle(f'{benchmark_name} {"epoch" if x_axis == "epoch" else "accuracy"}')

    plt.gca().legend(lines, labels)

    filename = filename.replace('@', 'at')
    plt.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_ood_specific(
    benchmark_name,
    nc_split='val',
    acc_split='val',
    ood_metric='AUROC',
):
    ood_tasks = benchmark2oodtasks[benchmark_name]
    _, epochs, acc_val, acc_train, _, *ood_dicts = load_benchmark_data(
        benchmark_name,
        nc_split=nc_split,
        ood_metric=ood_metric,
        ood_task=ood_tasks,
    )

    # Get ood values, mean over methods, delete unwanted methods
    ood_dicts = {task: val for task, val in zip(ood_tasks, ood_dicts)}
    for k, v in ood_dicts.items():
        for key in ['knn', 'mds', 'neco', 'nusa']:
            del v[key]
    ood_dicts = mean_over_inner_entries(ood_dicts)

    # Split into runs
    idx = np.where(epochs == 1)[0][1:]

    epochs = np.split(epochs, indices_or_sections=idx)
    epochs, _ = tolerant_mean(epochs)

    for k, v in ood_dicts.items():
        ood_dicts[k] = np.split(v, indices_or_sections=idx)

    if benchmark_name == 'imagenet':
        acc_val_mean = acc_val
        acc_train_mean = acc_train
    else:
        acc_val_split = np.split(acc_val, indices_or_sections=idx)
        acc_val_mean, _ = tolerant_mean(acc_val_split)
        acc_train_split = np.split(acc_train, indices_or_sections=idx)
        acc_train_mean, _ = tolerant_mean(acc_train_split)

    if acc_split == 'val':
        acc_mean = acc_val_mean
    elif acc_split == 'train':
        acc_mean = acc_train_mean

    ood_mean = {}
    ood_std = {}
    for k, v in ood_dicts.items():
        ood_mean[k], ood_std[k] = tolerant_mean(v)

    save_dir = path.res_plots / 'ood_specific'
    save_dir.mkdir(parents=True, exist_ok=True)

    data = (epochs, acc_mean, ood_mean, ood_std)

    _plot_specific(
        benchmark_name,
        save_dir,
        f'ood_{benchmark_name}_{ood_metric}_acc',
        ood_mean.keys(),
        data,
        None,
        acc_split,
        ood_metric,
        acc_train=acc_train_mean,
        acc_val=acc_val_mean,
    )
    _plot_specific(
        benchmark_name,
        save_dir,
        f'ood_{benchmark_name}_{ood_metric}_epoch',
        ood_mean.keys(),
        data,
        'epoch',
        acc_split,
        ood_metric,
        acc_train=acc_train_mean,
        acc_val=acc_val_mean,
    )


if __name__ == '__main__':
    ood_metric = 'AUROC'
    # ood_metric = 'FPR@95'

    # plot_ood('cifar10', ood_metric=ood_metric)
    plot_ood('cifar100', ood_metric=ood_metric)
    # plot_ood('imagenet200', ood_metric=ood_metric)
    # plot_ood('imagenet', ood_metric=ood_metric)
    # plot_ood('alexnet', ood_metric=ood_metric)
    # plot_ood('vgg', ood_metric=ood_metric)
    # plot_ood('mobilenet', ood_metric=ood_metric)

    # plot_ood_specific('cifar10', ood_metric=ood_metric)
    plot_ood_specific('cifar100', ood_metric=ood_metric)
    # plot_ood_specific('imagenet200', ood_metric=ood_metric)
    # plot_ood_specific('imagenet', ood_metric=ood_metric)
    # plot_ood_specific('alexnet', ood_metric=ood_metric)
    # plot_ood_specific('vgg', ood_metric=ood_metric)
    # plot_ood_specific('mobilenet', ood_metric=ood_metric)
