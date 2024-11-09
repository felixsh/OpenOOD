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


def nc_all_ckpt(benchmark_name, run_id):
    ckpt_dir = path.ckpt_root / benchmark_name / run_id
    ckpt_list = list(ckpt_dir.glob(ckpt_glob))

    for ckpt_path in ckpt_list:
        metrics = eval_nc(benchmark_name, ckpt_path)

        epoch_id = get_epoch_id(ckpt_path)
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
    ckpt_list = list(ckpt_dir.glob(ckpt_glob))

    for ckpt_path in ckpt_list:
        metrics, scores = eval_ood(benchmark_name, ckpt_path, postprocessor_name)

        epoch_id = get_epoch_id(ckpt_path)
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
