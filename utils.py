import re

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

    net.load_state_dict(load(ckpt_path, weights_only=True))
    net.cuda()
    net.eval()

    return net


def get_batch_size(benchmark_name):
    if benchmark_name == 'cifar10':
        batch_size = 1024
    elif benchmark_name == 'cifar100':
        batch_size = 1024
    elif benchmark_name == 'imagenet200':
        batch_size = 200
    elif benchmark_name == 'imagenet':
        batch_size = 100
    return batch_size


def get_episode(ckpt_path):
    """Return episode in format 'e100' """
    episode_name = f'e{re.search(r"_e(\d+)", ckpt_path).group(1)}'
    return episode_name
