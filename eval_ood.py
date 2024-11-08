from openood.evaluation_api import Evaluator
import path
from utils import load_network


def eval_ood(benchmark_name, postprocessor_name, ckpt_path):
    
    net = load_network(benchmark_name, ckpt_path)

    evaluator = Evaluator(
        net,
        id_name=benchmark_name,                # the target ID dataset
        data_root=str(path.data_root),         # change if necessary
        config_root=None,                      # see notes above
        preprocessor=None,                     # default preprocessing for the target ID dataset
        postprocessor_name=postprocessor_name, # the postprocessor to use
        postprocessor=None,                    # if you want to use your own postprocessor
        batch_size=1024,                       # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=8)                         # could use more num_workers outside colab

    ood_metrics = evaluator.eval_ood(fsood=False)

    return ood_metrics
