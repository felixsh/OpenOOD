import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import path
from utils import load_network, get_batch_size


class EvalAcc(object):
    def __init__(self, benchmark_name, ckpt_path) -> None:
        self.benchmark_name = benchmark_name
        self.ckpt_path = ckpt_path

        # Parameters
        batch_size = get_batch_size(benchmark_name)
        shuffle = False
        num_workers = 4

        # Prepare data
        data_root = str(path.data_root)
        preprocessor = get_default_preprocessor(benchmark_name)
        data_setup(data_root, benchmark_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        self.dataloader_dict = get_id_ood_dataloader(benchmark_name, data_root,
                                                    preprocessor, **loader_kwargs)

        from pprint import pprint
        pprint(self.dataloader_dict)
        input()
        
        # Load model
        self.net = load_network(benchmark_name, ckpt_path)

        # Get features, labels, logits, predictions
        exampel_batch = next(iter(self.dataloader_dict['id']['train']))
        exampel_inp = exampel_batch['data'].cuda().float()

        example_out = self.net(exampel_inp, return_feature=False)

        self.num_classes = example_out.shape[1]

    def compute(self, dataloader):
        accuracy = MulticlassAccuracy(num_classes=self.num_classes)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                inp = batch['data'].cuda().float()
                out = self.net(inp, return_feature=False)

                labels = batch['label']
                preds = out.argmax(dim=1).cpu().numpy()
                accuracy.update(preds, labels)
        
        return accuracy.compute().item()

    def walk(self, d: dict) -> dict:
        """Recursively walks through a dictionary and evaluates DataLoader objects."""
        result = {}
        for key, value in d.items():
            if isinstance(value, DataLoader):
                result[key] = self.compute(value)
            elif isinstance(value, dict):
                result[key] = self.walk(value)
            else:
                result[key] = None  # Placeholder for non-DataLoader items
        return result


def eval_acc(benchmark_name, ckpt_path):
    evaluator = EvalAcc(benchmark_name, ckpt_path)
    return evaluator.walk()


if __name__ == '__main__':
     benchmark_name = 'imagenet'
     eval_acc(benchmark_name, '')
