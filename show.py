from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate


data_path = Path('./res_data')
plot_path = Path('./res_plots')
plot_path.mkdir(exist_ok=True, parents=True)
marker_size = 144./plt.gcf().dpi


def get_data_id(filename):
    return '_'.join(str(filename.stem).split('_')[:-1])


def concat(arrays1d):
    return np.concatenate([a.reshape(-1, 1) for a in arrays1d], axis=1)


def kde2d(kde, X, Y):
    s0, s1 = X.shape
    Z = kde.cdf(concat((X.flatten(), Y.flatten())))
    return Z.reshape(s0, s1)


def plot_scores2d(data0, data1):
    f_id_0, f_ood_0, label0 = data0
    f_id_1, f_ood_1, label1 = data1

    id_data_id = get_data_id(f_id_0)
    id_score_0 = np.load(f_id_0)
    id_score_1 = np.load(f_id_1)

    data = concat((id_score_0, id_score_1))
    kde = KDEMultivariate(data, var_type='cc')

    for f0, f1 in zip(f_ood_0, f_ood_1):
        ood_data_id = get_data_id(f0)
        title = f'{ood_data_id}_{label0}_{label1}'
        ood_score_0 = np.load(f0)
        ood_score_1 = np.load(f1)
    
        plt.plot(ood_score_0, ood_score_1, '.', ms=marker_size, label=ood_data_id)
        plt.plot(id_score_0, id_score_1, '.', ms=marker_size, label=id_data_id)

        xmin, xmax, ymin, ymax = plt.axis()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        Z = kde2d(kde, X, Y)
        plt.contour(X, Y, Z, [0.01, 0.1, 0.5, 0.9, 0.99])

        plt.title(title)
        plt.xlabel(label0)
        plt.ylabel(label1)
        plt.legend()

        filename = f'{title}.png'
        plt.savefig(plot_path / filename)
        plt.close()


id_clst_file = list(data_path.glob('id*empiricalcovar.npy'))[0]
id_dist_file = list(data_path.glob('id*2-norm.npy'))[0]
id_coss_file = list(data_path.glob('id*cossim.npy'))[0]

ood_clst_files = sorted(list(data_path.glob('ood*empiricalcovar.npy')))
ood_dist_files = sorted(list(data_path.glob('ood*2-norm.npy')))
ood_coss_files = sorted(list(data_path.glob('ood*cossim.npy')))

clst_data = (id_clst_file, ood_clst_files, 'empcovar')
dist_data = (id_dist_file, ood_dist_files, '2norm')
coss_data = (id_coss_file, ood_coss_files, 'cossim')

plot_scores2d(clst_data, dist_data)
plot_scores2d(clst_data, coss_data)
plot_scores2d(coss_data, dist_data)
