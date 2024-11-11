import re

import numpy as np
from torch import load

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50


def load_network(benchmark_name, ckpt_path):
    if benchmark_name == 'cifar10':
        net = ResNet18_32x32(num_classes=10)
    elif benchmark_name == 'cifar100':
        net = ResNet18_32x32(num_classes=100)
    elif benchmark_name == 'imagenet200':
        net = ResNet18_224x224(num_classes=200)
    elif benchmark_name == 'imagenet':
        net = ResNet50(num_classes=1000)

    net.load_state_dict(load(ckpt_path, weights_only=True, map_location='cuda:0'))
    net.cuda()
    net.eval()

    return net


def get_batch_size(benchmark_name):
    # For 12GB VRAM
    if benchmark_name == 'cifar10':
        batch_size = 1024
    elif benchmark_name == 'cifar100':
        batch_size = 1024
    elif benchmark_name == 'imagenet200':
        batch_size = 512
    elif benchmark_name == 'imagenet':
        batch_size = 512
    return batch_size


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
