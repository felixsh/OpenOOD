import json

from omegaconf import OmegaConf
from pandas import HDFStore

from eval_nc import eval_nc
from eval_ood import eval_ood
import path
from utils import get_epoch_id, convert_numpy_to_lists


ckpt_glob = '*.ckpt'

postprocessors = [
    'msp',
    'odin',
    'mds',
    'react',
    'dice',
    'knn',
    'nusa',
    'vim',
    'ncscore',
    'neco',
    'epa'
]


def save_metrics(df, save_dir, key):
    # Print markdown table to file
    with open(save_dir / 'metrics.md', 'a') as f:
        f.write(f'---\n{key}\n')
        f.write(df.to_markdown())
        f.write('\n')

    # Store in HDF5 format
    with HDFStore(save_dir / 'metrics.h5') as store:
        store.put(key, df)


def save_scores(score_dict, save_dir, filename):
    save_dir.mkdir(exist_ok=True, parents=True)

    new_dict = {
        'id': score_dict['id'],
        'ood': score_dict['ood'],
    }
    converted_dict = convert_numpy_to_lists(new_dict)

    with open(save_dir / f'{filename}.json', 'w') as f:
        json.dump(converted_dict, f, indent=4)


def eval_postprocessor(benchmark_name, postprocessor_name):

    if benchmark_name == 'cifar10':
        ckpt_path = './results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    elif benchmark_name == 'cifar100':
        ckpt_path = './results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    elif benchmark_name == 'imagenet200':
        ckpt_path = './results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt'
    elif benchmark_name == 'imagenet':
        ckpt_path = './results/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt'

    # nc_metrics = eval_nc(benchmark_name, ckpt_path)
    ood_metrics, ood_scores = eval_ood(benchmark_name, ckpt_path, postprocessor_name)

    save_metrics(ood_metrics,
                 path.res_data,
                 benchmark_name,
                 postprocessor_name)
    
    save_scores(ood_scores,
                path.res_data,
                benchmark_name)


def nc_all_ckpt(benchmark_name, run_id):
    ckpt_dir = path.ckpt_root / benchmark_name / run_id
    ckpt_list = list(ckpt_dir.glob(ckpt_glob))

    for ckpt_path in ckpt_list:
        metrics = eval_nc(benchmark_name, ckpt_path)

        epoch_id = get_epoch_id(ckpt_path)
        save_dir = path.res_data / benchmark_name / run_id / epoch_id
        save_dir.mkdir(exist_ok=True, parents=True)

        save_metrics(metrics,
                     save_dir,
                     'nc')


def nc_best_ckpt(benchmark_name):
    ckpt_path = path.ckpt_root / benchmark_name / 'best.ckpt'

    save_dir = path.res_data / benchmark_name / 'best'
    save_dir.mkdir(exist_ok=True, parents=True)

    metrics = eval_nc(benchmark_name, ckpt_path)

    save_metrics(metrics,
                 save_dir,
                 'nc')


def ood_all_ckpt(benchmark_name, run_id, postprocessor_name):
    ckpt_dir = path.ckpt_root / benchmark_name / run_id
    ckpt_list = list(ckpt_dir.glob(ckpt_glob))

    for ckpt_path in ckpt_list:
        metrics, scores = eval_ood(benchmark_name, ckpt_path, postprocessor_name)

        epoch_id = get_epoch_id(ckpt_path)
        save_dir = path.res_data / benchmark_name / run_id / epoch_id
        save_dir.mkdir(exist_ok=True, parents=True)

        save_metrics(metrics,
                     save_dir,
                     postprocessor_name)

        save_scores(scores,
                    save_dir,
                    postprocessor_name)


def ood_best_ckpt(benchmark_name, postprocessor_name):
    ckpt_path = path.ckpt_root / benchmark_name / 'best.ckpt'

    save_dir = path.res_data / benchmark_name / 'best'
    save_dir.mkdir(exist_ok=True, parents=True)

    metrics, scores = eval_ood(benchmark_name, ckpt_path, postprocessor_name)

    save_metrics(metrics,
                 save_dir,
                 postprocessor_name)

    save_scores(scores,
                save_dir,
                postprocessor_name)


def eval_benchmark(benchmark_name, run_id):
    if run_id == 'best':
        nc_best_ckpt(benchmark_name)
    else:
        nc_all_ckpt(benchmark_name, run_id)

    for postpro in postprocessors:
        if run_id == 'best':
            ood_best_ckpt(benchmark_name, postpro)
        else:
            ood_all_ckpt(benchmark_name, run_id, postpro)


if __name__ == '__main__':
    # main_cfg = OmegaConf.load('cfg/main.yaml')
    cfg = OmegaConf.from_cli()
    # cfg = OmegaConf.merge(main_cfg, cli_cfg)

    # postprocessor options:
    # ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"]

    ood_all_ckpt('cifar10', 'run0', 'msp')
    nc_all_ckpt('cifar10', 'run0')

    ood_all_ckpt('cifar10', 'run0', 'knn')
    nc_all_ckpt('cifar10', 'run0')

    # eval_benchmark(cfg.benchmark, cfg.run)
