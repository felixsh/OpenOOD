from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import nc_toolbox as nctb

from .base_postprocessor import BasePostprocessor


class NCPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NCPostprocessor, self).__init__(config)
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
            
            self.mu_c = nctb.class_embedding_means(h, l)
            self.var_c = nctb.class_embedding_variances(h, l, self.mu_c)
            self.mu_g = nctb.global_embedding_mean(h)

            # w, b = net.get_fc()

            self.setup_flag = True
        else:
            pass
    
    def nc(self, h):
        # TODO implement stuff, h is batched [bs x d]
        return torch.max(h, dim=1)[0]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)
        scores = self.nc(features)
        _, preds = torch.max(logits, dim=1)
        return preds, scores

    def set_hyperparam(self, hyperparam: list):
        self.p = hyperparam[0]

    def get_hyperparam(self):
        return self.p
