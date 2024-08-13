from typing import Any

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NECOPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NECOPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        print('postprocessor.args:', self.args)
        self.d = self.args.d
        self.pca = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
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
            self.pca = PCA(n_components=self.d)
            self.pca.fit(h)

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)  # (b x c), (b x d)
        _, preds = torch.max(logits, dim=1)

        features_transformed = torch.as_tensor(self.pca.transform(features.cpu().numpy())).cuda()
        scores = torch.norm(features_transformed, dim=1) / torch.norm(features, dim=1)

        return preds, scores

    def set_hyperparam(self, hyperparam: list):
        self.d = hyperparam[0]

    def get_hyperparam(self):
        return self.d
