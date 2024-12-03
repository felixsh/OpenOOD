import json
from pathlib import Path
import re
from timeit import default_timer as timer
from datetime import timedelta

from natsort import natsorted
from omegaconf import OmegaConf
from pandas import HDFStore
import torch.multiprocessing as mp

from eval_nc import eval_nc
from eval_ood import eval_ood
from feature_cache import FeatureCache
import path
from utils import get_epoch_number, get_epoch_name, convert_numpy_to_lists, get_benchmark_name


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


def save_ood(df, save_dir, filename, key):
    # Store in HDF5 format
    with HDFStore(save_dir / f'{filename}.h5') as store:
        store.put(key, df)

    # Print markdown table to file
    with open(save_dir / f'{filename}.md', 'a') as f:
        f.write(f'---\n{key}\n')
        f.write(df.to_markdown())
        f.write('\n')


def save_nc(df, save_dir, filename, key):
    # Store in HDF5 format
    with HDFStore(save_dir / f'{filename}.h5') as store:
        store.put(key, df)

    # Print values to file
    with open(save_dir / f'{filename}.md', 'a') as f:
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


def existing_keys(save_dir, filename):
    full_name = save_dir / f'{filename}.h5'
    
    if full_name.is_file():
        with HDFStore(full_name, mode='r') as store:
            return list(store.keys())
    else:
        return []


def filter_ckpts(ckpt_list, filter_list=[1, 2, 5, 10, 20, 50, 100, 200, 500]):
    """Only use ckpts from epochs defined in filter list, plus final epoch"""
    # filter_list = [f-1 for f in filter_list]  # Shifted indices
    ckpt_list = natsorted(ckpt_list, key=str)
    ckpts_filtered = [p for p in ckpt_list if get_epoch_number(p) in filter_list]
    ckpts_filtered.append(ckpt_list[-1])
    ckpts_filtered = natsorted(list(set(ckpts_filtered)), key=str)
    return ckpts_filtered        


def eval_run(run_dir, ood_method_list=postprocessors):
    run_dir = Path(run_dir)
    ckpt_list = [p for p in run_dir.glob('*') if p.suffix in ckpt_suffixes]
    ckpt_list = filter_ckpts(ckpt_list)

    benchmark_name = get_benchmark_name(run_dir)
    print('BENCHMARK', benchmark_name)
    save_dir = path.res_data / run_dir.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)

    for ckpt_path in ckpt_list:
        start_time = timer()
        eval_ckpt(benchmark_name, ckpt_path, save_dir, list(ood_method_list))
        print(f'{'\033[91m'}Checkpoint took {timedelta(seconds=timer()-start_time)}{'\033[0m'}')


def eval_ckpt(benchmark_name, ckpt_path, save_dir, ood_method_list):
    file_name = get_epoch_name(ckpt_path)

    done_keys = existing_keys(save_dir, file_name).remove('nc')
    all_done = all([k in done_keys for k in ood_method_list + ['nc_train', 'nc_val']])

    if not all_done:
        feature_cache = FeatureCache(benchmark_name, ckpt_path)

        if not 'nc_train' in done_keys:
            nc_metrics = eval_nc(feature_cache, split='train')
            save_nc(nc_metrics, save_dir, file_name, 'nc_train')
        
        if not 'nc_val' in done_keys:
            nc_metrics = eval_nc(feature_cache, split='val')
            save_nc(nc_metrics, save_dir, file_name, 'nc_val')

        for ood_method in ood_method_list:
            if ood_method in done_keys:
                continue

            ood_metrics, _ = eval_ood(benchmark_name, ckpt_path, ood_method, feature_cache)
            save_ood(ood_metrics, save_dir, file_name, ood_method)


def recompute_all(ood_method_list=postprocessors):
    cache_list = natsorted(list(path.cache_root.glob('**/*_train.npz')), key=str)
    ckpt_list = [path.ckpt_root / p.relative_to(path.cache_root) for p in cache_list]
    ckpt_list = [Path(re.sub('_train.npz$', '.pth', str(p))) for p in ckpt_list]

    for ckpt_path in ckpt_list:
        benchmark_name = get_benchmark_name(ckpt_path)
        save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
        save_dir.mkdir(exist_ok=True, parents=True)

        eval_ckpt(benchmark_name, ckpt_path, save_dir, ood_method_list)


if __name__ == '__main__':
    # https://stackoverflow.com/q/64654838
    # mp.set_start_method('spawn', force=True)
    mp.set_start_method('fork', force=True)

    # main_cfg = OmegaConf.load('cfg/main.yaml')
    cfg = OmegaConf.from_cli()
    # cfg = OmegaConf.merge(main_cfg, cli_cfg)

    eval_run(cfg.run, cfg.ood)
