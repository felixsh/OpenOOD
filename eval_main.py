import json

from omegaconf import OmegaConf
from pandas import HDFStore

from eval_nc import eval_nc
from eval_ood import eval_ood
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
    filter_list = [f-1 for f in filter_list]  # Shifted indices
    ckpt_list = sorted(ckpt_list)
    ckpts_filtered = [p for p in ckpt_list if get_epoch_number(p) in filter_list]
    ckpts_filtered.append(ckpt_list[-1])
    ckpts_filtered = sorted(list(set(ckpts_filtered)))
    return ckpts_filtered


def nc_all_ckpt(benchmark_name, run_id):
    ckpt_dir = path.ckpt_root / benchmark_name / run_id
    ckpt_list = [p for p in ckpt_dir.glob('*') if p.suffix in ckpt_suffixes]
    ckpt_list = filter_ckpts(ckpt_list)

    for ckpt_path in ckpt_list:
        metrics = eval_nc(benchmark_name, ckpt_path)

        epoch_id = get_epoch_name(ckpt_path)
        save_dir = path.res_data / benchmark_name / run_id / epoch_id
        save_dir.mkdir(exist_ok=True, parents=True)

        save_nc(metrics,
                save_dir,
                'nc')


def nc_best_ckpt(benchmark_name):
    ckpt_path = path.ckpt_root / benchmark_name / 'best.ckpt'

    save_dir = path.res_data / benchmark_name / 'best'
    save_dir.mkdir(exist_ok=True, parents=True)

    metrics = eval_nc(benchmark_name, ckpt_path)

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


def ood_best_ckpt(benchmark_name, postprocessor_name):
    ckpt_path = path.ckpt_root / benchmark_name / 'best.ckpt'

    save_dir = path.res_data / benchmark_name / 'best'
    save_dir.mkdir(exist_ok=True, parents=True)

    metrics, scores = eval_ood(benchmark_name, ckpt_path, postprocessor_name)

    save_ood(metrics,
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
        try:
            if run_id == 'best':
                ood_best_ckpt(benchmark_name, postpro)
            else:
                ood_all_ckpt(benchmark_name, run_id, postpro)
        except Exception:
            continue


if __name__ == '__main__':
    # main_cfg = OmegaConf.load('cfg/main.yaml')
    cfg = OmegaConf.from_cli()
    # cfg = OmegaConf.merge(main_cfg, cli_cfg)

    eval_benchmark(cfg.benchmark, cfg.run)
