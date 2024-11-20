import re

import numpy as np
from torch import load

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50
import path


def load_network(benchmark_name, ckpt_path):

    if benchmark_name == 'cifar10_noise':
        net = ResNet18_32x32(num_classes=10)
        state_dict = load(ckpt_path, weights_only=True, map_location='cuda:0')
        state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
        state_dict.pop('extraction_layer.weight', None)
        state_dict.pop('extraction_layer.bias', None)
        net.load_state_dict(state_dict)
        net.cuda()
        net.eval()
        return net

    if benchmark_name == 'cifar10':
        net = ResNet18_32x32(num_classes=10)
    elif benchmark_name == 'cifar100':
        net = ResNet18_32x32(num_classes=100)
    elif benchmark_name == 'imagenet200':
        print('Load network', ckpt_path)
        if '_e150_' in str(ckpt_path):
            num_classes = 1000
            limit_classes = 200
        else:
            num_classes = 200
            limit_classes = None
        net = ResNet18_224x224(num_classes=num_classes, limit_classes=limit_classes)
    elif benchmark_name == 'imagenet':
        net = ResNet50(num_classes=1000)

    net.load_state_dict(load(ckpt_path, weights_only=True, map_location='cuda:0'))
    net.cuda()
    net.eval()

    return net


def get_batch_size(benchmark_name):
    # For 12GB VRAM
    if benchmark_name == 'cifar10':
        return 1024
    elif benchmark_name == 'cifar100':
        return 1024
    elif benchmark_name == 'imagenet200':
        return 256
    elif benchmark_name == 'imagenet':
        return 256
    else:
        raise NotImplementedError


def get_epoch_number(ckpt_path):
    """Return episode in format '100' """
    episode_number = int(re.search(r"_e(\d+)", str(ckpt_path.name)).group(1))
    return episode_number


def get_epoch_name(ckpt_path):
    """Return episode in format 'e100' """
    episode_name = f'e{get_epoch_number(ckpt_path)}'
    return episode_name


def convert_numpy_to_lists(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    else:
        return data  # Leave other data types unchanged


def convert_lists_to_numpy(data):
    if isinstance(data, dict):
        return {key: convert_lists_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Convert list to numpy array if it contains only numbers or nested lists
        try:
            return np.array(data)
        except ValueError:
            # In case it's a list of mixed types, keep it as a list
            return [convert_lists_to_numpy(item) for item in data]
    else:
        return data  # Leave other data types unchanged


def get_benchmark_name(full_path):
    rel_path = full_path.relative_to(path.ckpt_root)
    return str(rel_path.parents[-2])


if __name__ == '__main__':
    from pathlib import Path
    p = Path('/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_14-12_51_57')
    print(get_benchmark_name(p))
