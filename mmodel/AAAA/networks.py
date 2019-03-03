from torch import nn
import torch
from mmodel.basic_module import WeightedModule


from torchvision.models import resnet50


class FeatureExtroctor(WeightedModule):
    
    def __init__(self, params):
        super().__init__()

        res = resnet50(pretrained=True)
        layers = list(res.children())[:-2]

        self.F = torch.nn.Sequential(*layers)
        self.has_init = True
    
    def forward(self, inputs):
        features = self.F(inputs)
        return features

    def output_size(self):
        return [2048, 7, 7]


class Classifier(WeightedModule):
    def __init__(self, class_number):
        super().__init__()

        self.classifer = nn.Sequential(
            nn.Linear(2048, class_number),
            nn.Softmax(dim=1),
        )
 
    def forward(self, inputs):
        b = inputs.size()[0]
        feature = inputs.view(b, -1)
        predict = self.classifer(feature)
        return predict 

class DomainClassifier(WeightedModule):
    def __init__(self):
        super().__init__()

        self.classify = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


    def forward(self, inputs):
        b = inputs.size()[0]
        predict = self.classify(inputs.view(b, -1))
        return predict


    def get_output_shape(self):
        return (1024,1,1)


class SmallDomainClassifer(WeightedModule):

    def __init__(self):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b = inputs.size()[0]
        predict = self.classify(inputs.view(b, -1))
        return predict


if __name__ == "__main__":

    x = torch.Tensor(3, 10, 5, 5).random_(0, 10)
    # spool = nn.AdaptiveAvgPool2d(1)
    # def cpool():
    #     pool = nn.AdaptiveAvgPool1d(1)
    #     b, c, w, h = x.size()
    #     x = x.view(b, c, w*h).permute(0, 2, 1)
    #     x = pool(x).view(b, 1, w, h)
    #     return x


    # print(cpool(x).size())
