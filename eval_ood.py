from pathlib import Path

from pandas import DataFrame

import path
import utils
from feature_cache import FeatureCache
from openood.evaluation_api import Evaluator


def eval_ood(
    benchmark_name: str,
    ckpt_path: Path,
    postprocessor_name: str,
    feature_cache: FeatureCache,
) -> tuple[DataFrame, int | float | tuple[int, float] | None]:
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

    metrics = evaluator.eval_ood(fsood=False)

    try:
        hyperparam = evaluator.postprocessor.get_hyperparam()
    except AttributeError:
        hyperparam = None

    return metrics, hyperparam
