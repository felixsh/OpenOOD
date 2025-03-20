from pathlib import Path
from typing import Any

from filelock import FileLock
from pandas import DataFrame

import path
from feature_cache import FeatureCache
from openood.evaluation_api import Evaluator
from utils import get_batch_size, load_network


def eval_ood(
    benchmark_name: str,
    ckpt_path: Path,
    postprocessor_name: str,
    feature_cache: FeatureCache,
) -> tuple[DataFrame, dict[str, Any]]:
    network_name = benchmark_name
    if benchmark_name == 'cifar10_noise':
        benchmark_name = 'cifar10'

    net = load_network(network_name, ckpt_path)
    batch_size = get_batch_size(benchmark_name)

    evaluator = Evaluator(
        net,
        id_name=benchmark_name,  # the target ID dataset
        data_root=str(path.data_root),  # change if necessary
        config_root=None,  # see notes above
        preprocessor=None,  # default preprocessing for the target ID dataset
        postprocessor_name=postprocessor_name,  # the postprocessor to use
        postprocessor=None,  # if you want to use your own postprocessor
        batch_size=batch_size,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=1,  # could use more num_workers outside colab
        feature_cache=feature_cache,
    )

    metrics, scores = evaluator.eval_ood(fsood=False)

    filename = 'hyperparam.log'
    lock = FileLock(filename + '.lock')
    try:
        hyperparam = evaluator.postprocessor.get_hyperparam()
        with lock:
            with open(filename, 'a') as f:
                f.write(
                    f'{benchmark_name}, {postprocessor_name}, {hyperparam}, {str(ckpt_path)}\n'
                )
    except AttributeError:
        pass

    return metrics, scores
