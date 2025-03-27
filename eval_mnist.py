import os
from pathlib import Path

import nc_toolbox as nctb
import numpy as np
import torch
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from tqdm import tqdm

import database
import path
import utils
from eval_nc import _eval_nc
from feature_cache import FeatureCache
from openood.evaluation_api.preprocessor import Convert, default_preprocessing_dict
from openood.networks import ResNet18_32x32


def _eval(
    benchmark_name: str, dataset: str, model: Module
) -> tuple[float, DataFrame, torch.Tensor, torch.Tensor]:
    config = default_preprocessing_dict[benchmark_name]
    pre_size = config['pre_size']
    img_size = config['img_size']
    normalization = config['normalization']

    transform = transforms.Compose(
        [
            Convert('RGB'),
            transforms.Resize(
                pre_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
    )

    if dataset == 'mnist':
        test_dataset = datasets.MNIST(
            root=path.torchvision_root,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset == 'svhn':
        test_dataset = datasets.SVHN(
            root=path.torchvision_root,
            split='test',
            download=True,
            transform=transform,
        )
    elif dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root=path.torchvision_root,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    test_loader = DataLoader(
        test_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=8,
    )

    model.eval()
    accuracy_metric = MulticlassAccuracy(num_classes=10)

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            device = next(model.parameters()).device
            images = images.to(device)
            labels = labels.to(device)

            outputs, features = model(images, return_feature=True)
            preds = outputs.argmax(dim=1)

            accuracy_metric.update(preds, labels)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    accuracy = accuracy_metric.compute().item()

    all_features = torch.cat(all_features, dim=0).numpy()  # (n_samples, n_features)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (n_samples,)

    weights, bias = model.get_fc()  # (c x d), (c,)

    nc_df = _eval_nc(all_features, all_labels, weights, bias)

    return float(accuracy), nc_df, all_features, all_labels


def eval_mnist(ckpt_path: Path, dataset_name: str) -> None:
    assert dataset_name in ['mnist', 'svhn']

    MAX_NUM_THREADS = 8
    os.environ['OMP_NUM_THREADS'] = str(MAX_NUM_THREADS)
    torch.set_num_threads(MAX_NUM_THREADS)

    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = utils.get_benchmark_name(ckpt_path)
    assert benchmark_name == 'cifar10'

    model = ResNet18_32x32(num_classes=10)
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location='cuda:0')
    )

    acc, nc, _, _ = _eval(benchmark_name, dataset_name, model)

    model_name = utils.get_model_name(ckpt_path)
    run_id = utils.extract_datetime_from_path(ckpt_path)
    epoch = utils.get_epoch_number(ckpt_path)
    split = 'val'

    database.store_acc(
        benchmark_name, model_name, run_id, epoch, dataset_name, split, acc
    )
    database.store_nc(
        benchmark_name, model_name, run_id, epoch, dataset_name, split, nc
    )


def cdnv(mu0: np.ndarray, mu1: np.ndarray, var0: float, var1: float) -> float:
    square_dist = np.square(mu0 - mu1).sum()
    cdnv = (var0 + var1) / (2 * square_dist)
    return cdnv


def cdnv_mean(mu0: np.ndarray, mu1: np.ndarray, var0: float, var1: float) -> float:
    square_dist = np.square(mu0 - mu1).sum()
    cdnv = (var0 + var1) / (2 * square_dist)
    return cdnv


def save_stats(ckpt_path: Path, dataset_name: str) -> None:
    assert dataset_name in ['mnist', 'svhn']

    MAX_NUM_THREADS = 8
    os.environ['OMP_NUM_THREADS'] = str(MAX_NUM_THREADS)
    torch.set_num_threads(MAX_NUM_THREADS)

    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = utils.get_benchmark_name(ckpt_path)
    assert benchmark_name == 'cifar10'

    model = ResNet18_32x32(num_classes=10)
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location='cuda:0')
    )

    model_name = utils.get_model_name(ckpt_path)
    run_id = utils.extract_datetime_from_path(ckpt_path)
    epoch = utils.get_epoch_number(ckpt_path)

    # OOD
    _, _, features, labels = _eval(benchmark_name, dataset_name, model)
    mu_c = nctb.class_embedding_means(features, labels)
    var_c = nctb.class_embedding_variances(features, labels, mu_c)
    mu_g = nctb.global_embedding_mean(features)
    var_g = nctb.global_embedding_variance(features, mu_g)

    C = mu_c.shape[0]
    database.store_stats(
        benchmark_name,
        model_name,
        run_id,
        epoch,
        dataset_name,
        'val',
        C,
        mu_g,
        var_g,
        mu_c,
        var_c,
    )

    # ID
    feature_cache = FeatureCache(benchmark_name, ckpt_path)
    features = feature_cache.get('val', 'features')
    labels = feature_cache.get('val', 'labels')
    mu_c = nctb.class_embedding_means(features, labels)
    var_c = nctb.class_embedding_variances(features, labels, mu_c)
    mu_g = nctb.global_embedding_mean(features)
    var_g = nctb.global_embedding_variance(features, mu_g)

    C = mu_c.shape[0]
    database.store_stats(
        benchmark_name,
        model_name,
        run_id,
        epoch,
        benchmark_name,
        'val',
        C,
        mu_g,
        var_g,
        mu_c,
        var_c,
    )


def extract_test_samples(ckpt_path: Path) -> None:
    MAX_NUM_THREADS = 8
    os.environ['OMP_NUM_THREADS'] = str(MAX_NUM_THREADS)
    torch.set_num_threads(MAX_NUM_THREADS)

    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = utils.get_benchmark_name(ckpt_path)
    assert benchmark_name == 'cifar10'

    model = ResNet18_32x32(num_classes=10)
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location='cuda:0')
    )

    _, _, H_cifar10, L_cifar10 = _eval(benchmark_name, 'cifar10', model)
    _, _, H_mnist, L_mnist = _eval(benchmark_name, 'mnist', model)
    _, _, H_svhn, L_svhn = _eval(benchmark_name, 'svhn', model)

    save_dir = path.res_data / 'test_mnist'
    save_dir.mkdir(exist_ok=True)
    np.savez(
        save_dir / 'test_mnist.npz',
        H_cifar10=H_cifar10,
        L_cifar10=L_cifar10,
        H_mnist=H_mnist,
        L_mnist=L_mnist,
        H_svhn=H_svhn,
        L_svhn=L_svhn,
    )


def load_test_samples() -> tuple[np.ndarray, ...]:
    save_file = path.res_data / 'test_mnist' / 'test_mnist.npz'
    data = np.load(save_file)

    H_cifar10 = data['H_cifar10']
    L_cifar10 = data['L_cifar10']
    H_mnist = data['H_mnist']
    L_mnist = data['L_mnist']
    H_svhn = data['H_svhn']
    L_svhn = data['L_svhn']

    return (
        H_cifar10,
        L_cifar10,
        H_mnist,
        L_mnist,
        H_svhn,
        L_svhn,
    )


if __name__ == '__main__':
    # Testing only
    p = Path(
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-20_32_01/ResNet18_32x32_e300_i0.pth'
    )
    # eval_mnist(p, 'svhn')

    extract_test_samples(p)
