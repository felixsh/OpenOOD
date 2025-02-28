from pathlib import Path

import matplotlib.pyplot as plt
from natsort import natsorted

import path
from plot_utils import benchmark2loaddirs, load_acc, load_acc_train


def _save_plot(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png')
    fig.savefig(save_path / f'{filename}.pdf')
    plt.close()


def plot_acc(benchmark_name):
    # Get run dirs
    main_dirs = benchmark2loaddirs[benchmark_name]
    main_dirs = [Path(p) for p in main_dirs]
    run_dirs = natsorted(
        [
            subdir
            for p in main_dirs
            if p.is_dir()
            for subdir in p.iterdir()
            if subdir.is_dir()
        ],
        key=str,
    )
    print('number of runs', len(run_dirs))

    save_dir = path.res_plots / 'acc'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    epochs_val = []
    acc_val = []
    epochs_train = []
    acc_train = []

    for run_dir in run_dirs:
        acc_val_ = load_acc(run_dir, benchmark=benchmark_name)
        acc_val.append(acc_val_['val']['values'])
        epochs_val.append(acc_val_['val']['epochs'])

        acc_train_, epochs_train_ = load_acc_train(
            run_dir, benchmark=benchmark_name, return_epochs=True
        )
        acc_train.append(acc_train_)
        epochs_train.append(epochs_train_)

        # For debugging purposes
        # print(f'{acc_val_["val"]["epochs"][0]}..{acc_val_["val"]["epochs"][-1]}, \t{epochs_train_[0]}..{epochs_train_[-1]}')

    alpha = 0.5
    fig = plt.figure()
    for ep, acc in zip(epochs_train, acc_train):
        plt.plot(ep, acc, '-', alpha=alpha, color='tab:blue', label='train')
    for ep, acc in zip(epochs_val, acc_val):
        plt.plot(ep, acc, '-', alpha=alpha, color='tab:orange', label='val')

    plt.title(f'{benchmark_name}, n={len(run_dirs)}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    # Only one label per line bundle
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = {k: v for k, v in zip(labels, handles)}
    plt.legend(labels.values(), labels.keys())

    _save_plot(fig, save_dir, f'acc_{benchmark_name}')


if __name__ == '__main__':
    plot_acc('cifar10')
    plot_acc('cifar100')
    plot_acc('imagenet200')
    plot_acc('imagenet')
    plot_acc('alexnet')
    plot_acc('vgg')
    plot_acc('mobilenet')
