import json
import re
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

import torch.multiprocessing as mp
from natsort import natsorted
from omegaconf import OmegaConf
from pandas import HDFStore

import database
import path
import utils
from eval_acc import eval_acc, eval_acc_cache
from eval_nc import eval_nc
from eval_ood import eval_ood
from feature_cache import FeatureCache

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
    'epa',
]

# postprocessor options:
# ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"]


def save_ood(df, save_dir, filename, key):
    # Store in HDF5 format
    full_path = save_dir / f'{filename}.h5'
    lock = utils.get_lockfile(full_path)
    with lock:
        with HDFStore(full_path) as store:
            store.put(key, df)

    # Print markdown table to file
    full_path = save_dir / f'{filename}.md'
    lock = utils.get_lockfile(full_path)
    with lock:
        with open(full_path, 'a') as f:
            f.write(f'---\n{key}\n')
            f.write(df.to_markdown())
            f.write('\n')


def save_nc(df, save_dir, filename, key):
    # Store in HDF5 format
    full_path = save_dir / f'{filename}.h5'
    lock = utils.get_lockfile(full_path)
    with lock:
        with HDFStore(full_path) as store:
            store.put(key, df)

    # Print markdown table to file
    full_path = save_dir / f'{filename}.md'
    lock = utils.get_lockfile(full_path)
    with lock:
        with open(full_path, 'a') as f:
            f.write(f'---\n{key}\n')
            f.write(df.transpose().to_markdown())
            f.write('\n')


def save_acc(df, save_dir, filename):
    key = 'acc'

    # Store in HDF5 format
    full_path = save_dir / f'{filename}.h5'
    lock = utils.get_lockfile(full_path)
    with lock:
        with HDFStore(full_path) as store:
            store.put(key, df)

    # Print markdown table to file
    full_path = save_dir / f'{filename}.md'
    lock = utils.get_lockfile(full_path)
    with lock:
        with open(full_path, 'a') as f:
            f.write(f'---\n{key}\n')
            f.write(df.transpose().to_markdown())
            f.write('\n')


def save_scores(score_dict, save_dir, filename):
    save_dir.mkdir(exist_ok=True, parents=True)

    new_dict = {
        'id': score_dict['id'],
        'ood': score_dict['ood'],
    }
    converted_dict = utils.convert_numpy_to_lists(new_dict)

    with open(save_dir / f'{filename}.json', 'w') as f:
        json.dump(converted_dict, f, indent=4)


def existing_keys(save_dir, filename):
    full_path = save_dir / f'{filename}.h5'
    lock = utils.get_lockfile(full_path)

    with lock:
        if full_path.is_file():
            with HDFStore(full_path, mode='r') as store:
                key_list = list(store.keys())

            # Remove '/' from beginning of str key
            key_list = [k[1:] for k in key_list]

            # Remove old 'nc' key if present, only use new 'nc_train', 'nc_val' keys
            try:
                key_list.remove('nc')
            except ValueError:
                pass

            return key_list

        else:
            return []


def filtering_ckpts(
    ckpt_list, filter_list=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
):
    """Only use ckpts from epochs defined in filter list, plus final epoch"""
    # filter_list = [f-1 for f in filter_list]  # Shifted indices
    ckpts_filtered = [p for p in ckpt_list if utils.get_epoch_number(p) in filter_list]
    ckpts_filtered.append(ckpt_list[-1])
    ckpts_filtered = natsorted(list(set(ckpts_filtered)), key=str)
    return ckpts_filtered


def eval_run(run_dir, ood_method_list=postprocessors):
    run_dir = Path(run_dir)
    ckpt_list = get_run_ckpts(run_dir)

    benchmark_name = utils.get_benchmark_name(run_dir)
    print('BENCHMARK', benchmark_name)
    save_dir = path.res_data / run_dir.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)

    for ckpt_path in ckpt_list:
        start_time = timer()
        eval_ckpt_nc(benchmark_name, ckpt_path, save_dir)
        eval_ckpt_ood(benchmark_name, ckpt_path, save_dir, list(ood_method_list))
        eval_ckpt_acc(ckpt_path)
        print(
            f'{"\033[91m"}Checkpoint took {timedelta(seconds=timer() - start_time)}{"\033[0m"}'
        )


def eval_ckpt_nc(benchmark_name, ckpt_path, save_dir, recompute=False):
    file_name = utils.get_epoch_name(ckpt_path)

    done_keys = existing_keys(save_dir, file_name)
    all_done = all([k in done_keys for k in ['nc_train', 'nc_val']])

    if not all_done or recompute:
        feature_cache = FeatureCache(benchmark_name, ckpt_path)

        if 'nc_train' not in done_keys or recompute:
            nc_metrics = eval_nc(feature_cache, split='train')
            save_nc(nc_metrics, save_dir, file_name, 'nc_train')

        if 'nc_val' not in done_keys or recompute:
            nc_metrics = eval_nc(feature_cache, split='val')
            save_nc(nc_metrics, save_dir, file_name, 'nc_val')


def eval_ckpt_acc_nc(benchmark_name: str, ckpt_path: Path) -> None:
    model_name = utils.get_model_name(ckpt_path)
    run_id = utils.extract_datetime_from_path(ckpt_path)
    epoch = utils.get_epoch_number(ckpt_path)
    dataset = benchmark_name

    # Get Data
    feature_cache = FeatureCache(benchmark_name, ckpt_path)

    # Calculate Accuracy
    acc_train = eval_acc_cache(feature_cache, split='train')
    database.store_acc(
        benchmark_name, model_name, run_id, epoch, dataset, 'train', acc_train
    )

    acc_val = eval_acc_cache(feature_cache, split='val')
    database.store_acc(
        benchmark_name, model_name, run_id, epoch, dataset, 'val', acc_val
    )

    # Calculate Neural Collapse
    nc_train = eval_nc(feature_cache, split='train')
    database.store_nc(
        benchmark_name, model_name, run_id, epoch, dataset, 'train', nc_train
    )

    nc_val = eval_nc(feature_cache, split='val')
    database.store_nc(benchmark_name, model_name, run_id, epoch, dataset, 'val', nc_val)


def eval_ckpt_ood(
    benchmark_name: str, ckpt_path: Path, ood_method_list: list[str]
) -> None:
    model_name = utils.get_model_name(ckpt_path)
    run_id = utils.extract_datetime_from_path(ckpt_path)
    epoch = utils.get_epoch_number(ckpt_path)

    feature_cache = FeatureCache(benchmark_name, ckpt_path)

    for ood_method in ood_method_list:
        ood_metrics, _ = eval_ood(benchmark_name, ckpt_path, ood_method, feature_cache)
        database.store_ood(
            benchmark_name, model_name, run_id, epoch, ood_method, ood_metrics
        )


def eval_ckpt_acc(ckpt_path):
    file_name = utils.get_epoch_name(ckpt_path)
    acc_metrics = eval_acc(ckpt_path)
    save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_acc(acc_metrics, save_dir, file_name)


def get_run_ckpts(run_dir, filtering=True):
    run_dir = Path(run_dir)
    ckpt_list = [p for p in run_dir.glob('*') if p.suffix in ckpt_suffixes]
    ckpt_list = natsorted(ckpt_list, key=str)
    if filtering:
        ckpt_list = filtering_ckpts(ckpt_list)
    return ckpt_list


def get_previous_ckpts():
    cache_list = natsorted(list(path.cache_root.glob('**/*_train.npz')), key=str)
    ckpt_list = [path.ckpt_root / p.relative_to(path.cache_root) for p in cache_list]
    ckpt_list = [Path(re.sub('_train.npz$', '.pth', str(p))) for p in ckpt_list]
    return natsorted(ckpt_list, key=str)


def recompute_all(ood_method_list=postprocessors):
    for ckpt_path in get_previous_ckpts():
        print(ckpt_path)
        benchmark_name = utils.get_benchmark_name(ckpt_path)
        save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
        save_dir.mkdir(exist_ok=True, parents=True)

        eval_ckpt_ood(
            benchmark_name, ckpt_path, save_dir, ood_method_list, recompute=True
        )


if __name__ == '__main__':
    # https://stackoverflow.com/q/64654838
    # mp.set_start_method('spawn', force=True)
    mp.set_start_method('fork', force=True)

    # main_cfg = OmegaConf.load('cfg/main.yaml')
    main_cfg = OmegaConf.create({'recompute': False})
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(main_cfg, cli_cfg)

    if cfg.recompute:
        recompute_all(ood_method_list=[])
    else:
        eval_run(cfg.run, cfg.ood)
