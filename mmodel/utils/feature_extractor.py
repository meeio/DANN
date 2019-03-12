import torch
from torch import nn
from mmodel.basic_module import WeightedModule
from torchvision.models import resnet50

class TenthGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0.1

def one_tenth_grad(x):
    return TenthGrad.apply(x)

class Res50FeatureExtroctor(WeightedModule):
    
    def __init__(self, params):
        super().__init__()

        res = resnet50(pretrained=True)
        layers = list(res.children())[:-2]

        self.F = torch.nn.Sequential(*layers)
        self.has_init = True
    
    def forward(self, inputs):
        features = self.F(inputs)
        features = one_tenth_grad(features)
        return features

    def output_size(self):
        return [2048, 7, 7]