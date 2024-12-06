from eval_main import eval_ckpt
from utils import get_benchmark_name
from omegaconf import OmegaConf
import path


def recompute(ckpt_path, method):
    benchmark_name = get_benchmark_name(ckpt_path)
    save_dir = path.res_data / ckpt_path.parent.relative_to(path.ckpt_root)
    save_dir.mkdir(exist_ok=True, parents=True)

    eval_ckpt(benchmark_name, ckpt_path, save_dir, [method])


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    recompute(cfg.ckpt, cfg.method)
