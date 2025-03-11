import json
import re
import sys

import numpy as np
from filelock import FileLock
from torch import load

import path
from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50

dummy = ResNet50()


def str_to_class(classname):
    """https://stackoverflow.com/a/1176180"""
    return getattr(sys.modules[__name__], classname)


def load_network(benchmark_name, ckpt_path):
    # Get model name
    json_file = ckpt_path.parent / 'data.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    model_name = data['metadata']['model']
    print('MODELNAME', model_name)

    # Get number of classes, limitatitions (optional)
    limit_classes = None
    if benchmark_name == 'cifar10':
        num_classes = 10
    elif benchmark_name == 'cifar100':
        num_classes = 100
    elif benchmark_name == 'imagenet200':
        if '_e150_' in str(ckpt_path):
            num_classes = 1000
            limit_classes = 200
        else:
            num_classes = 200
    elif benchmark_name == 'imagenet':
        num_classes = 1000

    # Create model, load checkpoint
    if model_name == 'NCResNet18_32x32' or model_name == 'type':
        if benchmark_name in ['noise', 'cifar100']:
            net = ResNet18_32x32(num_classes=num_classes)
        elif benchmark_name == 'imagenet200':
            net = ResNet18_224x224(num_classes=num_classes)

        state_dict = load(ckpt_path, weights_only=True, map_location='cuda:0')
        state_dict.pop('extraction_layer.weight', None)
        state_dict.pop('extraction_layer.bias', None)
        state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

    else:
        model_class = str_to_class(model_name)

        if limit_classes is None:
            net = model_class(num_classes=num_classes)
        else:
            net = model_class(num_classes=num_classes, limit_classes=limit_classes)

        net.load_state_dict(load(ckpt_path, weights_only=True, map_location='cuda:0'))

    net.name = model_name
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
        return 128
    else:
        raise NotImplementedError


def get_epoch_number(ckpt_path):
    """Return episode in format '100'"""
    episode_number = int(re.search(r'_e(\d+)', str(ckpt_path.name)).group(1))
    return episode_number


def get_epoch_name(ckpt_path):
    """Return episode in format 'e100'"""
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


def get_model_name(full_path):
    rel_path = full_path.relative_to(path.ckpt_root)
    return str(rel_path.parents[-2])


def get_lockfile(path):
    return FileLock(path.with_suffix(path.suffix + '.lock'))


def extract_datetime_from_path(full_path):
    """
    Extracts a date-time string of format 'YYYY_MM_DD-HH_MM_SS' from a given file path.
    """
    pattern = r'(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})'
    match = re.search(pattern, str(full_path))
    return match.group(1) if match else None


if __name__ == '__main__':
    from pathlib import Path

    p = Path(
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/imagenet200/type/no_noise/1000+_epochs/run_imagenet200-1000_e1000_2025_03_06-07_44_05/NCResNet18_224x224_e1_i0.pth'
    )

    net = load_network('imagenet200', p)
