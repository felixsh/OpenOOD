import matplotlib.pyplot as plt
import nc_toolbox as nctb
import numpy as np

import path
from eval_mnist import load_test_samples


def cossim(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return np.sum(a * b)


def bincount(x):
    """Freedman-Diaconis number of bins"""
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


def plot_hist(a, name):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.hist(a, bins=bincount(a), density=True, color=colors[0])
    plt.axvline(a.mean(), label='mean', color=colors[1])
    plt.axvline(np.median(a), label='median', color=colors[2])
    plt.axvline(a.max(), label='max', color=colors[3])

    plt.title(name)
    plt.xlabel('norm')
    plt.ylabel('rel freq')
    plt.legend()

    save_dir = path.res_plots / 'hist'
    save_dir.mkdir(exist_ok=True)

    plt.savefig(save_dir / f'{name}.png', bbox_inches='tight')
    plt.close()


(
    H_cifar10,
    L_cifar10,
    H_mnist,
    L_mnist,
    H_svhn,
    L_svhn,
) = load_test_samples()

mu_c_cifar10 = nctb.class_embedding_means(H_cifar10, L_cifar10)
mu_g_cifar10 = nctb.global_embedding_mean(H_cifar10)
mu_g_mnist = nctb.global_embedding_mean(H_mnist)
mu_g_svhn = nctb.global_embedding_mean(H_svhn)

print('norm mu_g cifar10\t', np.linalg.norm(mu_g_cifar10))
print('norm mu_g mnist  \t', np.linalg.norm(mu_g_mnist))
print('norm mu_g svhn   \t', np.linalg.norm(mu_g_svhn))

# plot_hist(np.linalg.norm(H_cifar10, axis=1), 'cifar10')
# plot_hist(np.linalg.norm(H_mnist, axis=1), 'mnist')
# plot_hist(np.linalg.norm(H_svhn, axis=1), 'svhn')

print('norm mu_g cifar10, mnist\t', np.linalg.norm(mu_g_cifar10 - mu_g_mnist))
print('norm mu_g cifar10, svhn \t', np.linalg.norm(mu_g_cifar10 - mu_g_svhn))
print('norm mu_g mnist, svhn   \t', np.linalg.norm(mu_g_mnist - mu_g_svhn))

print('cossim mu_g cifar10, mnist\t', cossim(mu_g_cifar10, mu_g_mnist))
print('cossim mu_g cifar10, svhn \t', cossim(mu_g_cifar10, mu_g_svhn))
print('cossim mu_g mnist, svhn   \t', cossim(mu_g_mnist, mu_g_svhn))
