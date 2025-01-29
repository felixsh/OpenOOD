from pathlib import Path

from eval_main import eval_ckpt_nc, eval_ckpt_ood
from utils import get_benchmark_name
from omegaconf import OmegaConf
import path


def recompute(ckpt_path, method, recompute=False):
    print(ckpt_path)
    ckpt_path = Path(ckpt_path)
    benchmark_name = get_benchmark_name(ckpt_path)
    save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)

    if method == 'nc':
        eval_ckpt_nc(benchmark_name, ckpt_path, save_dir, recompute=recompute)
    else:
        eval_ckpt_ood(benchmark_name, ckpt_path, save_dir, [method], recompute=recompute)


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    recompute(cfg.ckpt, cfg.method, recompute=True)
