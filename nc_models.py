import torch
import torch.nn as nn
import numpy as np
import torchvision
from datetime import datetime
import copy

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_32x32 import BasicBlock
from openood.networks.resnet18_224x224 import ResNet18_224x224
from openood.networks.resnet50 import ResNet50

'''class NCTensor(torch.Tensor):
    def __init__(self, x, nc_features, *args, **kwargs):
        self.nc_features = nc_features

    def __new__(cls, x, nc_features, *args, **kwargs):
        return super(NCTensor, cls).__new__(cls, x, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return super(NCTensor, self).clone(*args, **kwargs)

    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            return self
        return NCTensor(new_obj, self.nc_features)'''

class NCFeatureExtractor(nn.Module):
    def __init__(self, module):
        # black magic: we change our class to the class of the module we want to wrap, copy all its attributes, and insert our own forward method
        new_dict = module.__dict__.copy()
        old_forward = module.forward
        new_class_dict = module.__class__.__dict__.copy()
        new_class_dict['forward'] = copy.copy(self.forward)
        __class__ = type(module.__class__.__name__, (module.__class__, object), new_class_dict)
        self.__dict__ = new_dict
        self.__class__ = __class__
        self.old_forward = old_forward

        self.nc_features = None

    def forward(self, x):
        y = self.old_forward(x)
        self.nc_features = x.detach().clone()
        return y

class NCWrapper(nn.Module):
    def __init__(self, model, extraction_layer):
        super(NCWrapper, self).__init__()
        self.model = model
        self.extraction_layer = extraction_layer

    def forward(self, x, return_feature=False):
        y = self.model(x)
        if return_feature:
            return y, self.extraction_layer.nc_features
        return y

    def get_fc(self):
        fc = self.extraction_layer
        w = fc.weight.cpu().detach().numpy()
        b = fc.bias.cpu().detach().numpy()
        return w, b

    def forward_threshold(self, x, threshold):
        _ = self.model(x)
        feat = self.extraction_layer.nc_features
        feat = feat.clip(max=threshold)
        return self.extraction_layer(feat)


class NCVGG16(NCWrapper):
    def __init__(self, num_classes):
        # pass auf, vgg hat dropout layer!
        model = torchvision.models.vgg16(pretrained=False, num_classes=num_classes)
        model.classifier[6] = NCFeatureExtractor(model.classifier[6])
        super(NCVGG16, self).__init__(model, model.classifier[6])

class NCAlexNet(NCWrapper):
    def __init__(self, num_classes):
        # pass auf, alexnet hat dropout layer!
        model = torchvision.models.alexnet(pretrained=False, num_classes=num_classes)
        model.classifier[6] = NCFeatureExtractor(model.classifier[6])
        super(NCAlexNet, self).__init__(model, model.classifier[6])

    def forward(self, x, return_feature=False):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        return super().forward(x, return_feature)

class NCMobileNetV2(NCWrapper):
    def __init__(self, num_classes):
        # pass auf, mobilenetv2 hat dropout layer!
        model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        model.classifier[1] = NCFeatureExtractor(model.classifier[1])
        super(NCMobileNetV2, self).__init__(model, model.classifier[1])

class NCLessNet18(NCWrapper):
    def __init__(self, num_classes):
        model = ResNet18_32x32(num_classes=num_classes)
        block = BasicBlock
        model.in_planes = 64
        model.layer2 = model._make_layer(block, 128, 2, stride=1)
        model.in_planes = 128
        model.layer3 = model._make_layer(block, 64, 2, stride=1)
        model.in_planes = 64
        model.layer4 = model._make_layer(block, 64, 2, stride=1)
        model.in_planes = 64
        model.fc = nn.Linear(64 * block.expansion, num_classes)
        model.feature_size = 64 * block.expansion
        model.fc = NCFeatureExtractor(model.fc)
        super(NCLessNet18, self).__init__(model, model.fc)
