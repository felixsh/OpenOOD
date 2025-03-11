from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

import path
from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from utils import get_batch_size, get_benchmark_name, load_network


class EvalAcc(object):
    def __init__(self, benchmark_name, ckpt_path) -> None:
        self.benchmark_name = benchmark_name
        self.ckpt_path = ckpt_path

        # Parameters
        batch_size = get_batch_size(benchmark_name)
        shuffle = False
        num_workers = 8

        # Prepare data
        data_root = str(path.data_root)
        preprocessor = get_default_preprocessor(benchmark_name)
        data_setup(data_root, benchmark_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
        }
        self.dataloader_dict = get_id_ood_dataloader(
            benchmark_name, data_root, preprocessor, **loader_kwargs
        )

        # Load model
        self.net = load_network(benchmark_name, ckpt_path)

        # Get features, labels, logits, predictions
        exampel_batch = next(iter(self.dataloader_dict['id']['train']))
        exampel_inp = exampel_batch['data'].cuda().float()

        example_out = self.net(exampel_inp, return_feature=False)

        self.num_classes = example_out.shape[1]

    def compute(self, dataloader):
        accuracy = MulticlassAccuracy(num_classes=self.num_classes).cuda()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                inp = batch['data'].cuda().float()
                out = self.net(inp, return_feature=False)

                labels = batch['label'].cuda()
                preds = out.argmax(dim=1)
                accuracy.update(preds, labels)

        return accuracy.compute().item()

    def walk_(self, d: dict) -> dict:
        """Recursively walks through a dictionary and evaluate DataLoader objects."""
        result = {}
        for key, value in d.items():
            if isinstance(value, DataLoader):
                result[key] = self.compute(value)
            elif isinstance(value, dict):
                result[key] = self.walk_(value)
            else:
                result[key] = None  # Placeholder for non-DataLoader items
        return result

    def walk(self):
        return self.walk_(self.dataloader_dict)

    def eval(self, split):
        assert split in ['train', 'val']
        acc = self.compute(self.dataloader_dict['id'][split])
        res = {'id': {split: acc}}
        return res


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map('{0[1]}'.format)
    return df


def eval_acc(ckpt_path, split):
    benchmark_name = get_benchmark_name(ckpt_path)
    evaluator = EvalAcc(benchmark_name, Path(ckpt_path))
    res = evaluator.eval(split)
    return nested_dict_to_df(res)


if __name__ == '__main__':
    benchmark = 'cifar10'
    test_ckpt = Path(
        '/mrtstorage/users/truetsch/neural_collapse_runs/benchmarks/cifar10/NCResNet18_32x32/noise/300+_epochs/noise_random_e300_2024_11_15-03_02_22/NCResNet18_32x32_e100_i0.pth'
    )
    res = eval_acc(test_ckpt, 'val')
    print(res)
