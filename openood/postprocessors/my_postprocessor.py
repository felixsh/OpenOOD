from typing import Any

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import torch
import torch.nn as nn
from tqdm import tqdm

import nc_toolbox as nctb

from .base_postprocessor import BasePostprocessor

import matplotlib.pyplot as plt

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
            estimator = MinCovDet
            self.covars = [estimator().fit(x) for x in h_split.values()]

            cluster_scores = self._cluster_score(h)
            dist_scores = self._dist_score(h)
            data = self._concat((cluster_scores, dist_scores))
            self.data_kde = KDEMultivariate(data, var_type='cc')

            self.setup_flag = True
        else:
            pass

    def _concat(self, arrays1d):
        return np.concatenate([a.reshape(-1, 1) for a in arrays1d], axis=1)

    def _cluster_score(self, features):
        dists = [cov.mahalanobis(features) for cov in self.covars]
        dists = self._concat(dists)  # b x c
        scores = np.max(-dists, axis=1)  # b x 1, higher is better
        return scores

    def _dist_score(self, features):
        return np.linalg.norm(features, axis=1, ord=2)  # higher is better

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)
        
        features = features.detach().cpu().numpy()
        cluster_scores = self._cluster_score(features)
        dist_scores = self._dist_score(features)
        
        data = self._concat((cluster_scores, dist_scores))
        combined_scores = self.data_kde.cdf(data)

        _, preds = torch.max(logits, dim=1)
        return preds, torch.as_tensor(combined_scores)

    def set_hyperparam(self, hyperparam: list):
        self.p = hyperparam[0]

    def get_hyperparam(self):
        return self.p
