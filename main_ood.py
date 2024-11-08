import re

from omegaconf import OmegaConf
from pandas import HDFStore
import torch

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224

import path


def eval_postprocessor(benchmark_name, postprocessor_name):

    if benchmark_name == 'cifar10':
        ckpt_path = './results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
        net = ResNet18_32x32(num_classes=10)
        ckpt_name = 'resnet18_32x32_e100'
    elif benchmark_name == 'cifar100':
        ckpt_path = './results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
        net = ResNet18_32x32(num_classes=100)
        ckpt_name = 'resnet18_32x32_e100'
    elif benchmark_name == 'imagenet200':
        ckpt_path = './results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt'
        net = ResNet18_224x224(num_classes=200)
        ckpt_name = 'resnet18_224x224_e90'

    # Get episode name
    episode_name = f'e{re.search(r"_e(\d+)", ckpt_path).group(1)}'
    print(episode_name)

    net.load_state_dict(torch.load(ckpt_path, weights_only=True))
    net.cuda()
    net.eval()

    evaluator = Evaluator(
        net,
        id_name=benchmark_name,                # the target ID dataset
        data_root=str(path.data_root),         # change if necessary
        config_root=None,                      # see notes above
        preprocessor=None,                     # default preprocessing for the target ID dataset
        postprocessor_name=postprocessor_name, # the postprocessor to use
        postprocessor=None,                    # if you want to use your own postprocessor
        batch_size=200,                        # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=2)                         # could use more num_workers outside colab

    metrics = evaluator.eval_ood(fsood=False)

    save_path = path.res_data / benchmark_name

    # Print markdown table to file
    with open(path.res_data / f'{benchmark_name}.md', 'a') as f:
        f.write(f'---\n{postprocessor_name}\n')
        f.write(metrics.to_markdown())
        f.write('\n')

    # Store in HDF5 format
    with HDFStore(path.res_data / f'{benchmark_name}.h5') as store:
        store.put(postprocessor_name, metrics)


if __name__ == '__main__':
    # main_cfg = OmegaConf.load('cfg/main.yaml')
    cfg = OmegaConf.from_cli()
    # cfg = OmegaConf.merge(main_cfg, cli_cfg)
    # cfg.benchmark = 'imagenet200'
    # cfg.postprocessor = 'knn'

    # postprocessor options:
    # ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"]

    eval_postprocessor(cfg.benchmark, cfg.postprocessor)
