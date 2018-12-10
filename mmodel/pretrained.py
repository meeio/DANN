import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from mmodel.basic_module import WeightedModule


class ResNetFeatureExtrctor(WeightedModule):
    
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # return with feature shape [7,7,2048]
        layers = list(resnet.children())[:-2]
        
        self.feature = nn.Sequential(*layers)
        self.has_init = True

    def forward(self, inputs):
        return self.feature(inputs)

    def output_shape(self):
        return (2048, 7, 7)