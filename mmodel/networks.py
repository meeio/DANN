from torch import nn
import torch
from mmodel.basic_module import WeightedModule

class MaskingLayer(WeightedModule):
    
    def __init__(self, channel, spatial, s_reduction=3, c_reduction=16):
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
        return atten
    
    def spatial_mask(self, inputs):
        b, c, w, h = inputs.size()
        inputs = inputs.view(b,c,w*h).permute(0,2,1)
        pooled =  self.pool_1d(inputs).view(b, w*h)
        # after spatil pool with shape(c, w*h)

        atten = self.s_atten(pooled).view(b, 1, w, h)
        return atten

    def forward(self, inputs):
        c_mask = self.channel_mask(inputs)
        s_mask = self.spatial_mask(inputs)
        return c_mask, s_mask

class Classifer(WeightedModule):

    def __init__(self, params, in_dim):
        super().__init__()
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, params.bottle_neck),
            nn.Linear(params.bottle_neck, params.class_num)
        )

    def forward(self, inputs):
        predict = self.classifer(inputs)
        return predict

class DomainClassifer(WeightedModule):
    def __init__(self, param, in_dim):
        super().__init__()

        hidden_size = param.hidden_size
        self.predict = nn.Sequential(

            # temp bottleneck
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            #1
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            #2
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, coeff=0):
        # x = x * 1.0
        inputs.register_hook(lambda grad: grad * -1 * coeff)
        b = inputs.size()[0]
        domain = self.predict(inputs.view(b, -1))
        return domain

if __name__ == "__main__":
    
    x = torch.Tensor(3,10,5,5).random_(0,10)
    spool = nn.AdaptiveAvgPool2d(1)

    def cpool():
        pool = nn.AdaptiveAvgPool1d(1)
        b, c, w, h = x.size()
        x = x.view(b, c, w*h).permute(0, 2, 1)
        x = pool(x).view(b, 1, w, h)
        return x

    print(cpool(x))
    print(cpool(x).size())