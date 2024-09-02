from typing import Any

from scipy.linalg import orth
import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class NuSAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NuSAPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        print('postprocessor.args:', self.args)
        self.C = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            W, _ = net.get_fc()  # (c x d), (c,)
            print(f"==>> W.shape: {W.shape}")
            self.C = torch.as_tensor(orth(W.T)).cuda()
            print(f"==>> C.shape: {self.C.shape}")
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)  # (b x c), (b x d)
        _, preds = torch.max(logits, dim=1)

        features_transformed = torch.matmul(features, self.C)
        norm_transformed = torch.norm(features_transformed, dim=1)
        norm = torch.norm(features, dim=1)
        scores = norm_transformed / norm

        return preds, scores

    def set_hyperparam(self, hyperparam: list):
        self.d = hyperparam[0]

    def get_hyperparam(self):
        return self.d
