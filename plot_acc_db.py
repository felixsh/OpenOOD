import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import path
from database import DB_NAME


def fetch_all_runs_acc(benchmark, model, dataset, split):
    conn = sqlite3.connect(DB_NAME)

    query = """
    SELECT epoch, run, value
    FROM acc
    WHERE benchmark = ? AND model = ? AND dataset = ? AND split = ?
    ORDER BY epoch ASC;
    """
    df = pd.read_sql_query(query, conn, params=(benchmark, model, dataset, split))

    conn.close()
    return df


def plot_all_runs_with_mean(benchmark, model, dataset, split='val'):
    df = fetch_all_runs_acc(benchmark, model, dataset, split)

    # Pivot the dataframe to structure runs as separate columns
    df_pivot = df.pivot(index='epoch', columns='run', values='value')

    epochs = df_pivot.index.to_numpy()

    # Plot all runs with transparency
    fig = plt.figure(figsize=(8, 6))
    plt.axhline(
        y=0.1,
        color='k',
        linestyle='--',
        alpha=0.5,
        label='random',
    )

    for run in df_pivot.columns:
        plt.plot(
            epochs,
            df_pivot[run],
            alpha=0.2,
            color='tab:blue',
        )

    # Compute mean and standard error
    mean_acc = df_pivot.mean(axis=1)
    std_err = df_pivot.std(axis=1) / np.sqrt(df_pivot.shape[1])

    # Plot mean with standard error
    plt.errorbar(
        epochs,
        mean_acc,
        yerr=std_err,
        color='tab:orange',
        label='mean acc ± std',
    )

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(f'{model} on {dataset} {split}')
    plt.gca().set_xscale('log')
    plt.legend()

    save_dir = path.res_plots / 'acc_specific'
    save_dir.mkdir(exist_ok=True, parents=True)
    filename = f'{benchmark}_{dataset}_{split}.png'

    plt.tight_layout()
    fig.savefig(save_dir / filename)
    plt.close(fig)


if __name__ == '__main__':
    plot_all_runs_with_mean('cifar10', 'ResNet18_32x32', 'mnist')
    plot_all_runs_with_mean('cifar10', 'ResNet18_32x32', 'svhn')
