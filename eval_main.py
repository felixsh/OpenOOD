import re
from pathlib import Path

from natsort import natsorted

import database
import path
import utils
from eval_acc import eval_acc_cache
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


def filtering_ckpts(
    ckpt_list: list[str | Path],
    filter_list: list[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
) -> list[str]:
    """Only use ckpts from epochs defined in filter list, plus final epoch"""
    # filter_list = [f-1 for f in filter_list]  # Shifted indices
    ckpts_filtered = [p for p in ckpt_list if utils.get_epoch_number(p) in filter_list]
    ckpts_filtered.append(ckpt_list[-1])
    ckpts_filtered = natsorted(list(set(ckpts_filtered)), key=str)
    return ckpts_filtered


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
        ood_metrics, hyperparams = eval_ood(
            benchmark_name, ckpt_path, ood_method, feature_cache
        )
        database.store_ood(
            benchmark_name,
            model_name,
            run_id,
            epoch,
            ood_method,
            ood_metrics,
            hyperparams,
        )


def get_run_ckpts(run_dir: str | Path, filtering: bool = True) -> list[str]:
    run_dir = Path(run_dir)
    ckpt_list = [p for p in run_dir.glob('*') if p.suffix in ckpt_suffixes]
    ckpt_list = natsorted(ckpt_list, key=str)
    if filtering:
        ckpt_list = filtering_ckpts(ckpt_list)
    return ckpt_list


def get_previous_ckpts() -> list[Path]:
    cache_list = natsorted(list(path.cache_root.glob('**/*_train.npz')), key=str)
    ckpt_list = [path.ckpt_root / p.relative_to(path.cache_root) for p in cache_list]
    ckpt_list = [Path(re.sub('_train.npz$', '.pth', str(p))) for p in ckpt_list]
    return natsorted(ckpt_list, key=str)
