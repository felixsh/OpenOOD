import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from plot_utils import load_benchmark_data


def _plot(acc, nc, ood, ood_label, resolution=100, cuts=3):

    inp = np.stack([acc, ood, nc], axis=0).T
    kde = KDEMultivariate(inp, 'ccc')

    x_lin = np.linspace(acc.min(), acc.max(), resolution)
    y_lin = np.linspace(ood.min(), ood.max(), resolution)
    z_lin = np.linspace(nc.min(), nc.max(), cuts**2)

    X, Y = np.meshgrid(x_lin, y_lin)
    ones = np.ones_like(X)

    res = []
    for z_ in z_lin:
        Z = z_ * ones
        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        pdf = kde.pdf(points).reshape(X.shape)
        res.append(pdf)

    fig, axes = plt.subplots(cuts, cuts, figsize=(10, 10))

    for ax, pdf, z_ in zip(axes.ravel(), res, z_lin):
        ax.contourf(X, Y, pdf)
        ax.axis('off')
        ax.text(0.95, 0.05, f'{z_:.2f}', ha='right', va='top', transform=ax.transAxes)

    plt.tight_layout()
    # fig.suptitle(f'P(acc, ood | nc)')
    return fig


def _save(fig, save_path, filename):
    fig.savefig(save_path / f'{filename}.png', bbox_inches='tight')
    fig.savefig(save_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def plot_cut(benchmark_name,
             nc_metric='nc1_cdnv',
             ood_metric='AUROC',
             ):
    run_ids, epochs, acc, nc, nearood, farood, save_dir = load_benchmark_data(benchmark_name, nc_metric, ood_metric)
    fig = _plot(acc, nc, nearood, 'near')
    _save(fig, save_dir, f'cut_near_{ood_metric}_{nc_metric}')
    fig = _plot(acc, nc, farood, 'far')
    _save(fig, save_dir, f'cut_far_{ood_metric}_{nc_metric}')


if __name__ == '__main__':
    plot_cut('cifar10')
    plot_cut('cifar100')
    plot_cut('imagenet200')
