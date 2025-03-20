from pathlib import Path
from typing import Any

from pandas import DataFrame

import path
import utils
from feature_cache import FeatureCache
from openood.evaluation_api import Evaluator


def save_hyperparam(
    evaluator: Evaluator, benchmark_name: str, postprocessor_name: str, ckpt_path: Path
) -> None:
    filename = 'hyperparam.log'
    try:
        hyperparam = evaluator.postprocessor.get_hyperparam()
        lock = utils.get_lockfile(filename)
        with lock:
            with open(filename, 'a') as f:
                f.write(
                    f'{benchmark_name}, {postprocessor_name}, {hyperparam}, {str(ckpt_path)}\n'
                )
    except AttributeError:
        pass


def eval_ood(
    benchmark_name: str,
    ckpt_path: Path,
    postprocessor_name: str,
    feature_cache: FeatureCache,
) -> tuple[DataFrame, dict[str, Any]]:
    net = utils.load_network(benchmark_name, ckpt_path)
    batch_size = utils.get_batch_size(benchmark_name)

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

    save_hyperparam(evaluator, benchmark_name, postprocessor_name, ckpt_path)

    return metrics, scores
