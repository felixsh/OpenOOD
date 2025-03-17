import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from tqdm import tqdm

import path
from database import store_acc, store_nc
from eval_nc import _eval_nc
from openood.evaluation_api.preprocessor import Convert, default_preprocessing_dict
from openood.networks import ResNet18_32x32
from utils import (
    extract_datetime_from_path,
    get_benchmark_name,
    get_epoch_number,
    get_model_name,
)


def _eval(benchmark_name, dataset, model):
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

    return accuracy, nc_df


def evaluate_model_on_mnist(ckpt_path, dataset):
    assert dataset in ['mnist', 'svhn']

    MAX_NUM_THREADS = 8
    os.environ['OMP_NUM_THREADS'] = str(MAX_NUM_THREADS)
    torch.set_num_threads(MAX_NUM_THREADS)

    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = get_benchmark_name(ckpt_path)
    assert benchmark_name == 'cifar10'

    model = ResNet18_32x32(num_classes=10)
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True, map_location='cuda:0')
    )

    acc, nc = _eval(benchmark_name, dataset, model)

    model_name = get_model_name(ckpt_path)
    run_id = extract_datetime_from_path(ckpt_path)
    epoch = get_epoch_number(ckpt_path)
    split = 'val'

    store_acc(benchmark_name, model_name, run_id, epoch, dataset, split, acc)
    store_nc(benchmark_name, model_name, run_id, epoch, dataset, split, nc)


if __name__ == '__main__':
    # Testing only
    p = Path(
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_23_07/ResNet18_32x32_e0_i0.pth'
    )
    evaluate_model_on_mnist(p, 'svhn')
