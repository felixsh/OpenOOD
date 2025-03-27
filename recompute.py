import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eval_main import eval_ckpt_acc_nc, eval_ckpt_ood
from eval_mnist import save_stats
from utils import get_benchmark_name

MAX_NUM_THREADS = 4


def recompute(ckpt_path: str, method: str) -> None:
    os.environ['OMP_NUM_THREADS'] = str(MAX_NUM_THREADS)
    torch.set_num_threads(MAX_NUM_THREADS)

    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = get_benchmark_name(ckpt_path)

    if method == 'accnc':
        eval_ckpt_acc_nc(benchmark_name, ckpt_path)
    elif method in ['mnist', 'svhn']:
        # eval_mnist(ckpt_path, method)
        save_stats(ckpt_path, method)
    else:
        eval_ckpt_ood(benchmark_name, ckpt_path, [method])


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    recompute(cfg.ckpt, cfg.method)
