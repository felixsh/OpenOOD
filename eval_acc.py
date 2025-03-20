from pathlib import Path

import numpy as np

from feature_cache import FeatureCache


def eval_acc_cache(feature_cache: FeatureCache, split: str = 'train') -> float:
    labels = feature_cache.get(split, 'labels')
    preds = feature_cache.get(split, 'predictions')
    return float(np.mean(labels == preds))


if __name__ == '__main__':
    # Small test
    from feature_cache import FeatureCache

    benchmark = 'cifar10'
    test_ckpt = Path(
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/ResNet18_32x32/no_noise/300+_epochs/run_e300_2024_11_11-15_24_50/ResNet18_32x32_e10_i0.pth'
    )
    feature_cache = FeatureCache(benchmark, test_ckpt, recompute=False)
    res = eval_acc_cache(feature_cache, 'val')
    print(res)
