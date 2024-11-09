from openood.evaluation_api import Evaluator
import path
from utils import load_network, get_batch_size


def eval_ood(benchmark_name, ckpt_path, postprocessor_name):
    
    net = load_network(benchmark_name, ckpt_path)
    batch_size = get_batch_size(benchmark_name)

    evaluator = Evaluator(
        net,
        id_name=benchmark_name,                # the target ID dataset
        data_root=str(path.data_root),         # change if necessary
        config_root=None,                      # see notes above
        preprocessor=None,                     # default preprocessing for the target ID dataset
        postprocessor_name=postprocessor_name, # the postprocessor to use
        postprocessor=None,                    # if you want to use your own postprocessor
        batch_size=batch_size,                 # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=0)                         # could use more num_workers outside colab

    metrics, scores  = evaluator.eval_ood(fsood=False)

    return metrics, scores
