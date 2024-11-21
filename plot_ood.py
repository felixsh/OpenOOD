from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import path
from plot_utils import colors, metric_markers
from plot_utils import load_acc, load_ood


benchmark2loaddirs = {
    'cifar10': (
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCAlexNet/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCMobileNetV2/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCResNet18_32x32/noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/NCVGG16/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/cifar10/ResNet18_32x32/no_noise/300+_epochs',
    ),
    'cifar100': (
        '/mrtstorage/users/hauser/openood_res/data/cifar100/ResNet18_32x32/no_noise/1000+_epochs',
    ),
    'imagenet200': (
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/150+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/200+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/300+_epochs',
        '/mrtstorage/users/hauser/openood_res/data/imagenet200/ResNet18_224x224/no_noise/400+_epochs',
    ),
    'imagenet': None,
}


def _plot(save_path, filename, ood_keys, data, far, x_axis, acc_split, ood_metric):
    epochs, avg_acc, avg_near_ood, avg_far_ood = data

    # Plotting
    # plt.title(f'{benchmark_name} {"far" if far else "near"}')
    for ood_key in ood_keys:
        y_values = avg_far_ood[ood_key] if far else avg_near_ood[ood_key]

        zipped =  zip(epochs, y_values) if x_axis == "epoch" else zip(avg_acc, y_values)

        # Remove NaNs from data
        valid_data = [(x, y) for x, y in zipped if not np.isnan(y)]
        if not valid_data:
            continue
        x_values, y_values = zip(*valid_data)
        plt.plot(x_values, y_values, '-', alpha=0.3, color=colors[1] if far else colors[0])
        plt.plot(x_values, y_values, metric_markers[ood_key],
                 color=colors[1] if far else colors[0], label=ood_key)

    if x_axis == "epoch":
        plt.xlabel('epoch')
        plt.gca().set_xscale('log')
    else:
        plt.xlabel(f'acc {acc_split}')
    plt.ylabel(ood_metric)

    # Create legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=2)

    plt.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    plt.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_acc_ood_avg(benchmark_name,
                     acc_split='val',  # 'train' or 'val'
                     ood_metric='AUROC',
                     ):
    # Get run dirs
    main_dirs = benchmark2loaddirs[benchmark_name]
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = [subdir for p in main_dirs if p.is_dir() for subdir in p.iterdir() if subdir.is_dir()]
    save_dir = path.res_plots / main_dirs[0].relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    acc_dict = defaultdict(list)
    nearood_dict = defaultdict(lambda: defaultdict(list))  # ood_key -> epoch -> list of values
    farood_dict = defaultdict(lambda: defaultdict(list))

    for run_dir in run_dirs:
        nearood, farood, epochs = load_ood(run_dir, ood_metric=ood_metric)

        for k in nearood.keys():
            for e, n, f in zip(epochs, nearood[k], farood[k]):
                nearood_dict[k][e].append(n)
                farood_dict[k][e].append(f)

        acc = load_acc(run_dir, filter_epochs=epochs)
        for e, a in zip(epochs, acc[acc_split]['values']):
            acc_dict[e].append(a)

    # Compute statistics
    epochs = list(acc_dict.keys())
    avg_acc = [np.mean(acc_dict[e]) for e in epochs]

    ood_keys = nearood_dict.keys()
    avg_near_ood = {
        k: [np.mean(nearood_dict[k][epoch]) if epoch in nearood_dict[k] else np.nan for epoch in epochs]
        for k in ood_keys
    }
    avg_far_ood = {
        k: [np.mean(farood_dict[k][epoch]) if epoch in farood_dict[k] else np.nan for epoch in epochs]
        for k in ood_keys
    }
    data = (epochs, avg_acc, avg_near_ood, avg_far_ood)

    _plot(save_dir, f'avg_{ood_metric}_near_acc_{acc_split}', ood_keys, data, False, None, acc_split, ood_metric)
    _plot(save_dir, f'avg_{ood_metric}_far_acc_{acc_split}', ood_keys, data, True, None, acc_split, ood_metric)
    _plot(save_dir, f'avg_{ood_metric}_near_epoch', ood_keys, data, False, 'epoch', acc_split, ood_metric)
    _plot(save_dir, f'avg_{ood_metric}_far_epoch', ood_keys, data, True, 'epoch', acc_split, ood_metric)


if __name__ == '__main__':
    plot_acc_ood_avg('cifar10')
