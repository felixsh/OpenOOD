from typing import Any

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import torch
import torch.nn as nn
from tqdm import tqdm

import nc_toolbox as nctb

from .base_postprocessor import BasePostprocessor


class MyPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(MyPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.p = self.args.p
        self.mu_c = None
        self.var_c = None
        self.mu_g = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            h = []
            l = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    h.append(feature.data.cpu().numpy())

                    l.append(batch['label'])

            h = np.concatenate(h, axis=0)
            l = np.concatenate(l, axis=0)
            h_split = nctb.split_embeddings(h, l)

            estimator = EmpiricalCovariance
            # estimator = MinCovDet
            self.covars = [estimator().fit(x) for x in h_split.values()]
# 
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)

        # Cluster score
        features = features.detach().cpu().numpy()
        dists = [cov.mahalanobis(features) for cov in self.covars]
        dists = [d.reshape(-1, 1) for d in dists]
        dists = np.concatenate(dists, axis=1)  # b x c
        scores = np.max(-dists, axis=1)  # b x 1, higher is better

        # Distance score
        # scores = torch.linalg.vector_norm(features, dim=1, ord=2)  # higher is better

        _, preds = torch.max(logits, dim=1)
        return preds, torch.as_tensor(scores)

    def set_hyperparam(self, hyperparam: list):
        self.p = hyperparam[0]

    def get_hyperparam(self):
        return self.p
