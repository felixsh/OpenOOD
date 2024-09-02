from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
from tqdm import tqdm

from nc_toolbox import principal_decomp

from .base_postprocessor import BasePostprocessor


class EPAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(EPAPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        print('postprocessor.args:', self.args)
        self.d = self.args.d
        self.o_prime = None
        self.P = None
        self.beta = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
    
    def _entropy(self, logits):
        """Torch in, Numpy out"""
        probits = torch.nn.functional.softmax(logits, dim=1)
        return Categorical(probs = probits).entropy().cpu().numpy()

    def _principal_angle(self, features):
        """Numpy in, Numpy out"""
        features -= self.o_prime
        features_transformed = np.matmul(features, self.P.T)
        norm_transformed = np.linalg.norm(features_transformed, axis=1)
        norm = np.linalg.norm(features, axis=1)
        a = norm_transformed / norm
        return np.arccos(a)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:

            # Calc new origin
            w, b = net.get_fc()  # (c x d), (c,)
            self.o_prime = -np.linalg.pinv(w) @ b

            # Get logits & features
            logits = []
            H = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    logit, feature = net(data, return_feature=True)
                    logits.append(logit)
                    H.append(feature)

            logits = torch.cat(logits, dim=0)
            H = torch.cat(H, dim=0).cpu().numpy()

            '''
            # Show characterisitic lenghts of the geometry
            b_norm = np.linalg.norm(b)
            o_prime_norm = np.linalg.norm(self.o_prime)
            mu_g = features.mean(dim=0)
            mu_g_norm = torch.norm(mu_g).item()
            alpha = torch.norm(features - mu_g, dim=1).mean()
            
            #print(f"$\|b\| = {b_norm:.4}$")
            #print(f"$\|o'\| = {o_prime_norm:.4}$")
            #print(f"$\|\mu_g\| = {mu_g_norm:.4}$")
            #print(f"$\|\\alpha\| = {alpha:.4}$")
            '''
            self.P, _ = principal_decomp(H - self.o_prime, n_components=self.d, center=False)

            entropy = self._entropy(logits)
            principal_angle = self._principal_angle(H)
            self.beta = float(entropy.max() / principal_angle.min())

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)  # (b x c), (b x d)
        _, preds = torch.max(logits, dim=1)

        principal_angle = self._principal_angle(features.cpu().numpy())
        entropy = self._entropy(logits)
        scores = self.beta * principal_angle + entropy
        scores = -scores  # higher is better

        return preds, torch.as_tensor(scores)

    def set_hyperparam(self, hyperparam: list):
        self.d = hyperparam[0]

    def get_hyperparam(self):
        return self.d
