import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from mmodel.basic_module import WeightedModule, _basic_weights_init_helper


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


class AlexNetFeatureExtrctor(WeightedModule):
    
    def __init__(self):
        super().__init__()
        alexnet = models.alexnet(pretrained=True)
        # return with feature shape [7,7,2048]
        layers = list(alexnet.children())[0]
        
        self.feature = nn.Sequential(*layers)
        self.has_init = True

    def forward(self, inputs):
        return self.feature(inputs)

    def output_shape(self):
        return (256, 6, 6)

class AlexBottleNeck(WeightedModule):

    def __init__(self):
        super().__init__()
        alexnet = models.alexnet(pretrained=True)
        layers = list(alexnet.children())[1][:-1]

        self.extractor = nn.Sequential(*layers)

        self.has_init = True

    def forward(self, inputs):
        b,_,_,_ = inputs.size()
        feature = self.extractor(inputs.view(b,-1))
        return feature
    
    def output_shape(self):
        return (4096, 1, 1)