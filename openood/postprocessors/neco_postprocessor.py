from typing import Any

from nc_toolbox import principal_decomp
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NECOPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NECOPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        print('postprocessor.args:', self.args)
        self.D = self.args.D
        self.center = self.args.center
        self.P = None
        self.mean = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            H = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    H.append(feature.data)

            H = torch.cat(H, dim=0).cpu().numpy()

            if self.center:
                P, _, mean = principal_decomp(H, n_components=self.D, center=True)
                self.mean = torch.as_tensor(mean).cuda()  # (d,)
            else:
                P, _ = principal_decomp(H, n_components=self.D, center=False)
            self.P = torch.as_tensor(P).cuda()  # (D, d)

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net.forward(data, return_feature=True)  # (b x c), (b x d)
        _, preds = torch.max(logits, dim=1)

        # (b x d) @ (d, D) = (b x D)
        if self.center:
            features_transformed = torch.matmul(features - self.mean, self.P.T)
        else:
            features_transformed = torch.matmul(features, self.P.T)
        scores = torch.norm(features_transformed, dim=1) / torch.norm(features, dim=1)  # (b,)

        return preds, scores

    def set_hyperparam(self, hyperparam: list):
        self.D = hyperparam[0]

    def get_hyperparam(self):
        return self.D
