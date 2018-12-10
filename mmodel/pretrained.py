import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models


class ResNetFeatureExtrctor(nn.Module):
    
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # return with feature shape [7,7,1048]
        layers = list(resnet.children())[:-2]
        
        self.feature = nn.Sequential(layers)

    def forward(self, inputs):
        return self.feature(inputs)