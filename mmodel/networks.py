from torch import nn
import torch
from mmodel.basic_module import WeightedModule

class MaskingLayer(WeightedModule):
    
    def __init__(self, channel, spatial, s_reduction=2, c_reduction=4):
        super(MaskingLayer, self).__init__()

        self.pool_1d = nn.AdaptiveAvgPool1d(1)
        self.s_atten = nn.Sequential(
            nn.Linear(spatial, spatial // s_reduction),
            nn.ReLU(inplace=True),
            nn.Linear(spatial//s_reduction, spatial),
            nn.Sigmoid()
        )

        self.pool_2d = nn.AdaptiveAvgPool2d(1)
        self.c_atten = nn.Sequential(
            nn.Linear(channel, channel // c_reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // c_reduction, channel),
            nn.Sigmoid()
        )

    def channel_mask(self, inputs):
        b, c, _, _ = inputs.size()
        inputs = self.pool_2d(inputs).view(b, c)
        atten = self.c_atten(inputs).view(b, c, 1, 1)
        atten = atten.expend_as(inputs)
        return atten
    
    def spatial_mask(self, inputs):
        b, c, w, h = inputs.size()
        inputs = inputs.view(b,c,w*h).permute(0,2,1)
        pooled =  self.pool_1d(inputs).view(b, w*h)
        atten = self.s_atten(pooled).view(b, 1, w, h)
        atten = atten.expend_as(inputs)
        return atten

    def forward(self, inputs):
        c_mask = self.channel_mask(inputs)
        s_mask = self.spatial_mask(inputs)
        return c_mask, s_mask

class Classifier(WeightedModule):
    def __init__(self):
        super().__init__()

        self.feature_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.feature_fc = nn.Sequential(
            nn.Linear(5*5*128, 1024),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            nn.Linear(1024, 64),
            nn.ReLU(True),
        )

        self.feature_classif = nn.Sequential(
            nn.Linear(64, 10)
        )
    
    def forward(self, inputs):
        feature = self.feature_conv(inputs)
        b = feature.size()[0]
        feature = self.feature_fc(feature.view(b, -1))
        predict = self.feature_classif(feature)
        return feature, predict
    


if __name__ == "__main__":
    
    x = torch.Tensor(3,10,5,5).random_(0,10)
    # spool = nn.AdaptiveAvgPool2d(1)

    # def cpool():
    #     pool = nn.AdaptiveAvgPool1d(1)
    #     b, c, w, h = x.size()
    #     x = x.view(b, c, w*h).permute(0, 2, 1)
    #     x = pool(x).view(b, 1, w, h)
    #     return x                         

    # print(cpool(x))
    # print(cpool(x).size())