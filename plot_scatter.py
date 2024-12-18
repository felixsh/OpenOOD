from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import path
from plot_utils import colors, metric_markers
from plot_utils import load_acc, load_nc, load_ood


benchmark2loaddirs = {
    'cifar10': (
        #'/mrtstorage/users/hauser/openood_res/data/cifar10/NCAlexNet/no_noise/300+_epochs',
        #'/mrtstorage/users/hauser/openood_res/data/cifar10/NCMobileNetV2/no_noise/300+_epochs',
        #'/mrtstorage/users/hauser/openood_res/data/cifar10/NCVGG16/no_noise/300+_epochs',
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


def trim_arrays(dict1, dict2, array1, array2):
    min_length = min(
        min(arr.shape[0] for arr in dict1.values()),
        min(arr.shape[0] for arr in dict2.values())
    )
    
    trimmed_dict1 = {key: arr[:min_length] for key, arr in dict1.items()}
    trimmed_dict2 = {key: arr[:min_length] for key, arr in dict2.items()}
    
    trimmed_array1 = array1[:min_length]
    trimmed_array2 = array2[:min_length]
    
    return trimmed_dict1, trimmed_dict2, trimmed_array1, trimmed_array2


def load_benchmark_data(benchmark_name,
                        nc_metric='nc1_cdnv',
                        ood_metric='AUROC',
                        ):
    # Get run dirs
    main_dirs = benchmark2loaddirs[benchmark_name]
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = [subdir for p in main_dirs if p.is_dir() for subdir in p.iterdir() if subdir.is_dir()]
    save_dir = path.res_plots / main_dirs[0].relative_to(path.res_data).parents[-2]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    epochs = []
    run_ids = []
    data = []

    for run_id, run_dir in enumerate(run_dirs):
        nc, _ = load_nc(run_dir)
        nearood_dict, farood_dict, epochs_ = load_ood(run_dir, ood_metric=ood_metric)
        acc = load_acc(run_dir, filter_epochs=epochs_)

        nc = np.squeeze(nc[nc_metric])
        acc = acc['val']['values']

        # Needed if not all benchmarks are computed
        nearood_dict, farood_dict, nc, acc = trim_arrays(nearood_dict, farood_dict, nc, acc)
        
        nearood = np.mean([v for v in nearood_dict.values()], axis=0)
        farood = np.mean([v for v in farood_dict.values()], axis=0)

        for ep, a, n, nood, food in zip(epochs_, acc, nc, nearood, farood):
            data.append([a, n, nood, food])
            epochs.append(ep)
            run_ids.append(run_id)
    
    # [[acc, nc, nearood, farood]]
    data = np.array(data)

    run_ids = np.array(run_ids)
    epochs = np.array(epochs)

    acc = data[:, 0]
    nc = data[:, 1]
    nearood = data[:, 2]
    farood = data[:, 3]

    return run_ids, epochs, acc, nc, nearood, farood, save_dir


def _plot(acc, nc, ood, run_ids, nc_metric, ood_metric, ood_label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # c = [colors[i] for i in run_ids]
    c = run_ids

    axes.ravel()[0].scatter(acc, nc, c=c, marker='o')
    axes.ravel()[0].set_xlabel(f'acc val')
    axes.ravel()[0].set_ylabel(nc_metric)

    axes.ravel()[1].scatter(acc, ood, c=c, marker='o')
    axes.ravel()[1].set_xlabel(f'acc val')
    axes.ravel()[1].set_ylabel(f'{ood_metric} {ood_label}')

    axes.ravel()[2].scatter(nc, ood, c=c, marker='o')
    axes.ravel()[2].set_xlabel(nc_metric)
    axes.ravel()[2].set_ylabel(f'{ood_metric} {ood_label}')

    plt.tight_layout()
    return fig


def _save(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    fig.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_scatter(benchmark_name,
                 nc_metric='nc1_cdnv',
                 ood_metric='AUROC',
                 ):
    run_ids, epochs, acc, nc, nearood, farood, save_dir = load_benchmark_data(benchmark_name, nc_metric, ood_metric)

    # Epochs to color index
    _, color_id = np.unique(epochs, return_inverse=True)
    color_id += 1

    # Plot and save
    fig = _plot(acc, nc, nearood, color_id, nc_metric, ood_metric, 'near')
    _save(fig, save_dir, f'scatter_near_{nc_metric}_{ood_metric}')
    fig = _plot(acc, nc, farood, color_id, nc_metric, ood_metric, 'far')
    _save(fig, save_dir, f'scatter_far_{nc_metric}_{ood_metric}')


if __name__ == '__main__':
    plot_scatter('cifar100')