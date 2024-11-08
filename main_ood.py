from omegaconf import OmegaConf
from pandas import HDFStore

from eval_ood import eval_ood
import path


def eval_postprocessor(benchmark_name, postprocessor_name):

    if benchmark_name == 'cifar10':
        ckpt_path = './results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    elif benchmark_name == 'cifar100':
        ckpt_path = './results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    elif benchmark_name == 'imagenet200':
        ckpt_path = './results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt'
    elif benchmark_name == 'imagenet':
        ckpt_path = './results/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt'

    ood_metrics = eval_ood(benchmark_name, postprocessor_name, ckpt_path)

    # save_path = path.res_data / benchmark_name

    # Print markdown table to file
    with open(path.res_data / f'{benchmark_name}.md', 'a') as f:
        f.write(f'---\n{postprocessor_name}\n')
        f.write(ood_metrics.to_markdown())
        f.write('\n')

    # Store in HDF5 format
    with HDFStore(path.res_data / f'{benchmark_name}.h5') as store:
        store.put(postprocessor_name, ood_metrics)


if __name__ == '__main__':
    # main_cfg = OmegaConf.load('cfg/main.yaml')
    cfg = OmegaConf.from_cli()
    # cfg = OmegaConf.merge(main_cfg, cli_cfg)
    # cfg.benchmark = 'imagenet200'
    # cfg.postprocessor = 'knn'

    # postprocessor options:
    # ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"]

    eval_postprocessor(cfg.benchmark, cfg.postprocessor)
