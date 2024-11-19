from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import nc_toolbox as nctb

from .base_postprocessor import BasePostprocessor


class NCScorePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NCScorePostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        print('postprocessor.args:', self.args)
        self.alpha = self.args.alpha
        self.mu_g = None
        self.w = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, feature_cache=None):
        if not self.setup_flag:
            
            if feature_cache is None:
                h = []
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

                h = np.concatenate(h, axis=0)
            
            else:
                h = feature_cache.get('train', 'features')

            self.mu_g = torch.as_tensor(nctb.global_embedding_mean(h)).cuda()

            self.w, _ = net.get_fc()  # (c x d)
            self.w = torch.as_tensor(self.w).cuda()

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)
        _, preds = torch.max(logits, dim=1)

        h_cent = features - self.mu_g  # (b x d)
        h_cent /= torch.norm(h_cent, dim=1, keepdim=True)

        p_scores = torch.sum(h_cent * self.w[preds, :], dim=1)

        l1_scores = torch.norm(features, p=1, dim=1)
        scores = self.alpha * l1_scores + p_scores
        return preds, scores

    def set_hyperparam(self, hyperparam: list):
        self.alpha = hyperparam[0]

    def get_hyperparam(self):
        return self.alpha
