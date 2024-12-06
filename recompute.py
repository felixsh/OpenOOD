from pathlib import Path

from eval_main import eval_ckpt_ood
from utils import get_benchmark_name
from omegaconf import OmegaConf
import path


def recompute(ckpt_path, method):
    ckpt_path = Path(ckpt_path)
    benchmark_name = get_benchmark_name(ckpt_path)
    save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)

    eval_ckpt_ood(benchmark_name, ckpt_path, save_dir, [method], recompute=True)


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    recompute(cfg.ckpt, cfg.method)
