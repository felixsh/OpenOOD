from pathlib import Path

from omegaconf import OmegaConf

from eval_main import eval_ckpt_acc


def compute_acc_train(ckpt_path, i, n):
    print(f'{i}/{n}', ckpt_path)
    eval_ckpt_acc(Path(ckpt_path))


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    compute_acc_train(cfg.ckpt, cfg.i, cfg.n)