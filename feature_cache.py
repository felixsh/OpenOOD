from zipfile import BadZipFile

import nc_toolbox as nctb
import numpy as np
import torch
from tqdm import tqdm

import path
from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from utils import get_batch_size, load_network


class FeatureCache:
    def __init__(self, benchmark_name, ckpt_path, recompute=False):
        self.benchmark_name = benchmark_name
        self.cache_path = path.cache_root
        self.ckpt_path = ckpt_path
        self.recompute = recompute

        full_path = path.cache_root / ckpt_path.relative_to(path.ckpt_root)
        self.train_path = full_path.with_name(f'{full_path.stem}_train.npz')
        self.val_path = full_path.with_name(f'{full_path.stem}_val.npz')

        self.data = {}
        self.data['train'] = self._load_or_compute(self.train_path, split='train')
        self.data['val'] = self._load_or_compute(self.val_path, split='val')

    def get(self, split, key, return_torch=False):
        assert split in ('train', 'val')
        assert key in ('logits', 'features', 'labels', 'predictions', 'weights', 'bias')

        try:
            res = self.data[split][key]
        except BadZipFile:
            self.data[split] = self._load_or_compute(
                self.train_path, split=split, recompute=True
            )
            res = self.data[split][key]

        if return_torch:
            return torch.as_tensor(res)
        else:
            return res

    def _load_or_compute(self, data_path, split='train', recompute=False):
        try:
            if recompute:
                raise FileNotFoundError

            return np.load(data_path)

        except FileNotFoundError:
            logits, features, labels, predictions, weights, bias = self._compute(
                self.ckpt_path, split=split
            )

            data_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                data_path,
                logits=logits,
                features=features,
                labels=labels,
                predictions=predictions,
                weights=weights,
                bias=bias,
            )
            return np.load(data_path)

    def _compute(self, ckpt_path, split='train'):
        # Parameters
        batch_size = get_batch_size(self.benchmark_name)
        shuffle = False
        num_workers = 4

        # Prepare stuff
        data_root = str(path.data_root)
        preprocessor = get_default_preprocessor(self.benchmark_name)
        net = load_network(self.benchmark_name, ckpt_path)

        # Load data
        data_setup(data_root, self.benchmark_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
        }
        dataloader_dict = get_id_ood_dataloader(
            self.benchmark_name, data_root, preprocessor, **loader_kwargs
        )
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

    def mean_feature_norm(self, centered=False):
        H = self.data['val']['features']
        if centered:
            mu_g = nctb.global_embedding_mean(H)
            H = H - mu_g
        H_norm = np.linalg.norm(H, axis=1)
        return float(H_norm.mean())

    def mean_cluster_size(self):
        H = self.data['val']['features']
        L = self.data['val']['labels']
        mu_c = nctb.class_embedding_means(H, L)
        H_centered = nctb.center_embeddings(H, L, mu_c)
        H_norm = np.linalg.norm(H_centered, axis=1)
        return float(H_norm.mean())

    def mean_cluster_dist(self):
        H = self.data['val']['features']
        L = self.data['val']['labels']
        mu_c = nctb.class_embedding_means(H, L)
        C = mu_c.shape[0]
        idx0, idx1 = np.triu_indices(C, k=1)
        diff = mu_c[idx0] - mu_c[idx1]
        diff_norm = np.linalg.norm(diff, axis=1)
        assert diff.shape[1] == mu_c.shape[1]
        return float(diff_norm.mean())

    def mu_g_norm(self):
        H = self.data['val']['features']
        mu_g = nctb.global_embedding_mean(H)
        return float(np.linalg.norm(mu_g, ord=2))
