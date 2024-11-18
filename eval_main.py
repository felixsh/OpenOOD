import json

from omegaconf import OmegaConf
from pandas import HDFStore
import torch.multiprocessing as mp

from eval_nc import eval_nc
from eval_ood import eval_ood
from natsort import natsorted
import path
from utils import get_epoch_number, get_epoch_name, convert_numpy_to_lists


ckpt_suffixes = ['.ckpt', '.pth']

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

# postprocessor options:
# ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"]


def save_ood(df, save_dir, key):
    # Store in HDF5 format
    with HDFStore(save_dir / 'metrics.h5') as store:
        store.put(key, df)

    # Print markdown table to file
    with open(save_dir / 'metrics.md', 'a') as f:
        f.write(f'---\n{key}\n')
        f.write(df.to_markdown())
        f.write('\n')


def save_nc(df, save_dir, key):
    # Store in HDF5 format
    with HDFStore(save_dir / 'metrics.h5') as store:
        store.put(key, df)

    # Print values to file
    with open(save_dir / 'metrics.md', 'a') as f:
        f.write(f'---\n{key}\n')
        f.write(df.transpose().to_markdown())
        f.write('\n')


def save_scores(score_dict, save_dir, filename):
    save_dir.mkdir(exist_ok=True, parents=True)

    new_dict = {
        'id': score_dict['id'],
        'ood': score_dict['ood'],
    }
    converted_dict = convert_numpy_to_lists(new_dict)

    with open(save_dir / f'{filename}.json', 'w') as f:
        json.dump(converted_dict, f, indent=4)


def filter_ckpts(ckpt_list, filter_list=[1, 2, 5, 10, 20, 50, 100, 200, 500]):
    """Only use ckpts from epochs defined in filter list, plus final epoch"""
    # filter_list = [f-1 for f in filter_list]  # Shifted indices
    ckpt_list = natsorted(ckpt_list, key=str)
    ckpts_filtered = [p for p in ckpt_list if get_epoch_number(p) in filter_list]
    ckpts_filtered.append(ckpt_list[-1])
    ckpts_filtered = natsorted(list(set(ckpts_filtered)), key=str)
    return ckpts_filtered


def eval_nc(benchmark_name, run_id, epoch=None):
    ckpt_dir = path.ckpt_root / benchmark_name / run_id
    ckpt_list = [p for p in ckpt_dir.glob('*') if p.suffix in ckpt_suffixes]
    ckpt_list = filter_ckpts(ckpt_list)

    if epoch is not None:
        ckpt_list = [p for p in ckpt_list if f'e{epoch}' in str(p)]

    for ckpt_path in ckpt_list:
        metrics = eval_nc(benchmark_name, ckpt_path)

        epoch_id = get_epoch_name(ckpt_path)
        save_dir = path.res_data / benchmark_name / run_id / epoch_id
        save_dir.mkdir(exist_ok=True, parents=True)

        save_nc(metrics,
                save_dir,
                'nc')


def ood_all_ckpt(benchmark_name, run_id, postprocessor_name):
    ckpt_dir = path.ckpt_root / benchmark_name / run_id
    ckpt_list = [p for p in ckpt_dir.glob('*') if p.suffix in ckpt_suffixes]
    ckpt_list = filter_ckpts(ckpt_list)

    for ckpt_path in ckpt_list:
        metrics, scores = eval_ood(benchmark_name, ckpt_path, postprocessor_name)

        epoch_id = get_epoch_name(ckpt_path)
        save_dir = path.res_data / benchmark_name / run_id / epoch_id
        save_dir.mkdir(exist_ok=True, parents=True)

        save_ood(metrics,
                 save_dir,
                 postprocessor_name)

        save_scores(scores,
                    save_dir,
                    postprocessor_name)


def eval_benchmark(benchmark_name, run_id, pps):
    for postpro in pps:
        ood_all_ckpt(benchmark_name, run_id, postpro)


def eval_noise(benchmark_name, run_id):
    assert benchmark_name == 'cifar10_noise', 'Other not implemented'

    base_dir = path.ckpt_root / benchmark_name
    ckpt_dirs = [p for p in base_dir.glob('*') if p.is_dir()]

    for ckpt_dir in ckpt_dirs:
        ckpt_list = [p for p in ckpt_dir.glob('*') if p.suffix in ckpt_suffixes]
        ckpt_path = natsorted(ckpt_list, key=str)[-1]

        epoch_id = get_epoch_name(ckpt_path)
        save_dir = path.res_data / benchmark_name / run_id
        save_dir.mkdir(exist_ok=True, parents=True)
        
        nc_metrics = eval_nc(benchmark_name, ckpt_path)
        save_nc(nc_metrics, save_dir, 'nc')

        for postpro in postprocessors:
            ood_metrics, _ = eval_ood(benchmark_name, ckpt_path, postpro)
            save_ood(ood_metrics, save_dir, postpro)


if __name__ == '__main__':
    # https://stackoverflow.com/q/64654838
    # mp.set_start_method('spawn', force=True)
    mp.set_start_method('fork', force=True)

    # main_cfg = OmegaConf.load('cfg/main.yaml')
    cfg = OmegaConf.from_cli()
    # cfg = OmegaConf.merge(main_cfg, cli_cfg)

    if cfg.benchmark == 'cifar10_noise':
        eval_noise(cfg.benchmark, cfg.run)

    if 'pps' in cfg:
        eval_benchmark(cfg.benchmark, cfg.run, cfg.pps)
    else:
        eval_nc(cfg.benchmark, cfg.run, cfg.epoch)
