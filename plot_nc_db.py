import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

import path
from database import DB_NAME


def fetch_acc_from_db(benchmark, model, dataset, split):
    conn = sqlite3.connect(DB_NAME)

    acc_query = """
    SELECT epoch, AVG(value) AS mean_acc, 
           SQRT(AVG(value * value) - AVG(value) * AVG(value)) AS std_acc
    FROM acc
    WHERE benchmark = ? AND model = ? AND dataset = ? AND split = ?
    GROUP BY epoch
    ORDER BY epoch ASC;
    """
    acc_df = pd.read_sql_query(
        acc_query, conn, params=(benchmark, model, dataset, split)
    )
    conn.close()

    epochs = acc_df['epoch'].to_numpy()
    acc_mean = acc_df['mean_acc'].to_numpy()
    acc_std = acc_df['std_acc'].fillna(0).to_numpy()  # Handle NaN in standard deviation

    return epochs, acc_mean, acc_std


def fetch_nc_from_db(benchmark, model, dataset, split):
    conn = sqlite3.connect(DB_NAME)

    nc_query = """
    SELECT epoch, 
        AVG(nc1_weak_between) AS nc1_weak_between,
        SQRT(AVG(nc1_weak_between * nc1_weak_between) - AVG(nc1_weak_between) * AVG(nc1_weak_between)) AS nc1_weak_between_std,

        AVG(nc1_weak_within) AS nc1_weak_within,
        SQRT(AVG(nc1_weak_within * nc1_weak_within) - AVG(nc1_weak_within) * AVG(nc1_weak_within)) AS nc1_weak_within_std,

        AVG(nc1_cdnv_mean) AS nc1_cdnv_mean,
        SQRT(AVG(nc1_cdnv_mean * nc1_cdnv_mean) - AVG(nc1_cdnv_mean) * AVG(nc1_cdnv_mean)) AS nc1_cdnv_mean_std,

        AVG(nc2_equinormness_mean) AS nc2_equinormness_mean,
        SQRT(AVG(nc2_equinormness_mean * nc2_equinormness_mean) - AVG(nc2_equinormness_mean) * AVG(nc2_equinormness_mean)) AS nc2_equinormness_mean_std,

        AVG(nc2_equiangularity_mean) AS nc2_equiangularity_mean,
        SQRT(AVG(nc2_equiangularity_mean * nc2_equiangularity_mean) - AVG(nc2_equiangularity_mean) * AVG(nc2_equiangularity_mean)) AS nc2_equiangularity_mean_std,

        AVG(nc3_self_duality) AS nc3_self_duality,
        SQRT(AVG(nc3_self_duality * nc3_self_duality) - AVG(nc3_self_duality) * AVG(nc3_self_duality)) AS nc3_self_duality_std,

        AVG(unc3_uniform_duality_mean) AS unc3_uniform_duality_mean,
        SQRT(AVG(unc3_uniform_duality_mean * unc3_uniform_duality_mean) - AVG(unc3_uniform_duality_mean) * AVG(unc3_uniform_duality_mean)) AS unc3_uniform_duality_mean_std,

        AVG(nc4_classifier_agreement) AS nc4_classifier_agreement,
        SQRT(AVG(nc4_classifier_agreement * nc4_classifier_agreement) - AVG(nc4_classifier_agreement) * AVG(nc4_classifier_agreement)) AS nc4_classifier_agreement_std
    FROM nc
    WHERE benchmark = ? AND model = ? AND dataset = ? AND split = ?
    GROUP BY epoch
    ORDER BY epoch ASC;
    """
    nc_df = pd.read_sql_query(nc_query, conn, params=(benchmark, model, dataset, split))
    conn.close()

    return nc_df


def plot_nc_from_db(
    benchmark, model, dataset, compare_dataset, split='val', with_errorbars=True
):
    """
    Plots accuracy and NC metrics per epoch from SQLite database,
    using NC and accuracy from `dataset`, but only accuracy from `compare_dataset`.

    :param benchmark: The benchmark name.
    :param model: The model name.
    :param dataset: The dataset name (used for NC and accuracy).
    :param compare_dataset: The second dataset for accuracy comparison.
    :param split: The data split (default: "val").
    :param with_errorbars: Whether to include error bars.
    """
    epochs, acc_mean, acc_std = fetch_acc_from_db(benchmark, model, dataset, split)
    nc_df = fetch_nc_from_db(benchmark, model, dataset, split)

    # Fetch train/val accuracy from `compare_dataset`
    _, acc_train_mean, acc_train_std = fetch_acc_from_db(
        benchmark, model, compare_dataset, 'train'
    )
    # _, acc_val_mean, acc_val_std = fetch_acc_from_db(
    #     benchmark, model, compare_dataset, 'val'
    # )

    # Define subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax in axes.flatten():
        ax.set_xscale('log')

    # Subplot (0,0): nc1 metrics including `nc1_cdnv_mean`
    axes[0, 0].errorbar(
        epochs,
        nc_df['nc1_weak_between'],
        yerr=nc_df['nc1_weak_between_std'] if with_errorbars else None,
        label='nc1_weak_between',
        marker='o',
    )
    axes[0, 0].errorbar(
        epochs,
        nc_df['nc1_weak_within'],
        yerr=nc_df['nc1_weak_within_std'] if with_errorbars else None,
        label='nc1_weak_within',
        marker='s',
    )
    ax001 = axes[0, 0].twinx()
    ax001.errorbar(
        epochs,
        nc_df['nc1_cdnv_mean'],
        yerr=nc_df['nc1_cdnv_mean_std'] if with_errorbars else None,
        label='nc1_cdnv_mean',
        marker='D',
        color='tab:green',
    )
    axes[0, 0].set_ylabel('nc1_weak')
    ax001.set_ylabel('nc1_cdnv_mean')
    axes[0, 0].legend()
    ax001.legend()

    # Subplot (0,1): nc2 metrics
    axes[0, 1].errorbar(
        epochs,
        nc_df['nc2_equinormness_mean'],
        yerr=nc_df['nc2_equinormness_mean_std'] if with_errorbars else None,
        label='nc2_equinormness_mean',
        marker='o',
    )
    axes[0, 1].errorbar(
        epochs,
        nc_df['nc2_equiangularity_mean'],
        yerr=nc_df['nc2_equiangularity_mean_std'] if with_errorbars else None,
        label='nc2_equiangularity_mean',
        marker='s',
    )
    axes[0, 1].set_ylabel('nc2 metrics')
    axes[0, 1].legend()

    # Subplot (1,0): nc3 & unc3
    axes[1, 0].errorbar(
        epochs,
        nc_df['nc3_self_duality'],
        yerr=nc_df['nc3_self_duality_std'] if with_errorbars else None,
        label='nc3_self_duality',
        marker='o',
    )
    axes[1, 0].errorbar(
        epochs,
        nc_df['unc3_uniform_duality_mean'],
        yerr=nc_df['unc3_uniform_duality_mean_std'] if with_errorbars else None,
        label='unc3_uniform_duality_mean',
        marker='s',
    )
    axes[1, 0].set_ylabel('nc3 metrics')
    axes[1, 0].legend()

    # Subplot (1,1): Accuracy comparison with additional dataset
    axes[1, 1].errorbar(
        epochs,
        acc_mean,
        yerr=acc_std if with_errorbars else None,
        label=f'{dataset} {split} acc',
        marker='o',
    )
    axes[1, 1].errorbar(
        epochs,
        nc_df['nc4_classifier_agreement'],
        yerr=nc_df['nc4_classifier_agreement_std'] if with_errorbars else None,
        label='nc4_classifier_agreement',
        marker='s',
    )
    axes[1, 1].errorbar(
        epochs,
        acc_train_mean,
        yerr=acc_train_std if with_errorbars else None,
        label=f'{compare_dataset} train acc',
        marker='^',
    )
    # axes[1, 1].errorbar(
    #     epochs,
    #     acc_val_mean,
    #     yerr=acc_val_std if with_errorbars else None,
    #     label=f'{compare_dataset} Val Accuracy',
    #     marker='v',
    # )
    axes[1, 1].set_ylabel('accuracy / agreement')
    axes[1, 1].legend()

    # Set x-labels
    for ax in axes.flatten():
        ax.set_xlabel('epochs')

    # Save the plot
    filename = f'{benchmark}_{dataset}'
    save_dir = path.res_plots / 'nc_specific'
    save_dir.mkdir(exist_ok=True, parents=True)

    plt.tight_layout()
    fig.savefig(save_dir / f'{filename}.png')
    plt.close()


if __name__ == '__main__':
    plot_nc_from_db('cifar10', 'ResNet18_32x32', 'mnist', 'cifar10')
    plot_nc_from_db('cifar10', 'ResNet18_32x32', 'svhn', 'cifar10')
