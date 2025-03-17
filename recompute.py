import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

import path
from eval_main import eval_ckpt_acc, eval_ckpt_nc, eval_ckpt_ood
from eval_mnist import evaluate_model_on_mnist
from utils import get_benchmark_name

MAX_NUM_THREADS = 4


def recompute(ckpt_path, method, recompute=False):
    os.environ['OMP_NUM_THREADS'] = str(MAX_NUM_THREADS)
    torch.set_num_threads(MAX_NUM_THREADS)

    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = get_benchmark_name(ckpt_path)
    save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)

    if method == 'nc':
        eval_ckpt_nc(benchmark_name, ckpt_path, save_dir, recompute=recompute)
    elif method == 'acc':
        eval_ckpt_acc(ckpt_path)
    elif method in ['mnist', 'svhn']:
        evaluate_model_on_mnist(ckpt_path, method)
    else:
        eval_ckpt_ood(
            benchmark_name, ckpt_path, save_dir, [method], recompute=recompute
        )


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    recompute(cfg.ckpt, cfg.method, recompute=True)
