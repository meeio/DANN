from torch import nn
import torch


class SELayer(nn.Module):
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.atten = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CSELayer(nn.Module):

    def __init__(self, spatial, reduction):
        super(CSELayer, self).__init__()

        self.pool_1d = nn.AdaptiveAvgPool1d(1)
        self.atten = nn.Sequential(
            nn.Linear(spatial, spatial // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(spatial//reduction, spatial),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, w, h = inputs.size()
        inputs = inputs.view(b,c,w*h).permute(0,2,1)
        pooled =  self.pool_1d(inputs).view(b, w*h)
        # after spatil pool with shape(c, w*h)

        atten = self.atten(pooled).view(b, 1, w, h)
        return inputs * atten

if __name__ == "__main__":
    
    x = torch.Tensor(3,10,5,5).random_(0,10)
    spool = nn.AdaptiveAvgPool2d(1)

    def cpool(x):
        pool = nn.AdaptiveAvgPool1d(1)
        b, c, w, h = x.size()
        x = x.view(b, c, w*h).permute(0, 2, 1)
        x = pool(x).view(b, 1, w, h)
        return x

    print(cpool(x))
    print(cpool(x).size())