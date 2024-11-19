import numpy as np
import torch
from tqdm import tqdm

from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import path
from utils import load_network, get_batch_size


class FeatureCache():

    def __init__(self, benchmark_name, ckpt_path):
        self.benchmark_name = benchmark_name
        self.cache_path = path.cache_root
        self.ckpt_path = ckpt_path

        full_path = path.cache_root / ckpt_path.relative_to(path.ckpt_root)
        self.train_path = full_path.with_name(f'{full_path.stem}_train.npz')
        self.val_path = full_path.with_name(f'{full_path.stem}_val.npz')

        self.data = {}
        self.data['train'] = self._load_or_compute(self.train_path, split='train')
        self.data['val'] = self._load_or_compute(self.val_path, split='val')
    
    def get(self, split, key):
        assert split in ('train', 'val')
        assert key in ('logits', 'features', 'labels', 'predictions', 'weights', 'bias')
        return self.data[split][key]

    def _load_or_compute(self, data_path, split='train'):
        try:
            return np.load(data_path)
        except FileNotFoundError:
            logits, features, labels, predictions, weights, bias = self._compute(self.ckpt_path, split=split)
            
            data_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                data_path,
                logits=logits,
                features=features,
                labels=labels,
                predictions=predictions,
                weights=weights,
                bias=bias
            )
            return np.load(data_path)

    def _compute(self, ckpt_path, split='train'):

        # Parameters
        batch_size = get_batch_size(self.benchmark_name)
        shuffle = False
        num_workers = 6

        # Prepare stuff
        data_root = str(path.data_root)
        preprocessor = get_default_preprocessor(self.benchmark_name)
        net = load_network(self.benchmark_name, ckpt_path)

        # Load data
        data_setup(data_root, self.benchmark_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(self.benchmark_name, data_root,
                                                    preprocessor, **loader_kwargs)
        dataloader = dataloader_dict['id'][split]

        # Get features, labels, logits, predictions
        exampel_batch = next(iter(dataloader))
        exampel_inp = exampel_batch['data'].cuda().float()

        example_out, exampel_feature = net(exampel_inp, return_feature=True)

        n = len(dataloader.dataset)
        d = exampel_feature.shape[1]
        c = example_out.shape[1]
        assert exampel_feature.shape[0] == example_out.shape[0]

        outputs = np.empty((n, c), dtype=np.float32)
        features = np.empty((n, d), dtype=np.float32)
        labels = np.empty(n, dtype=int)
        predictions = np.empty(n, dtype=int)

        with torch.no_grad():
            idx = 0
            for batch in tqdm(dataloader, desc=split):
                inp = batch['data'].cuda().float()
                out, feature = net(inp, return_feature=True)

                bs = inp.shape[0]
                outputs[idx : idx + bs, :] = out.cpu().numpy()
                features[idx : idx + bs, :] = feature.flatten(start_dim=1).cpu().numpy()
                labels[idx : idx + bs] = batch['label']
                predictions[idx : idx + bs] = out.argmax(dim=1).cpu().numpy()
                idx += bs
        
        weights, bias = net.get_fc()  # (c x d), (c,)

        return outputs, features, labels, predictions, weights, bias
        