from torch import nn
import torch
from mmodel.basic_module import WeightedModule


class FeatureExtractor(WeightedModule):
    def __init__(self, params):
        super().__init__()

        img_c = 1 if params.gray else 3

        self.feature_conv = nn.Sequential(
            nn.Conv2d(img_c, 64, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


    def forward(self, inputs):
        feature = self.feature_conv(inputs)
        return feature

    def get_output_shape(self):
        return (128, 5, 5)

class Classifier(WeightedModule):
    def __init__(self):
        super().__init__()

        self.feature_fc = nn.Sequential(
            nn.Linear(5 * 5 * 128, 1024),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
        )

        self.predict = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b = inputs.size()[0]
        feature_fc = self.feature_fc(inputs.view(b, -1))
        predict = self.predict(feature_fc)
        return feature_fc, predict
        
    def get_output_shape(self):
        return (1024,1,1)


class DomainClassifier(WeightedModule):
    def __init__(self):
        super().__init__()

        self.feature_fc = nn.Sequential(
            nn.Linear(5 * 5 * 128, 1024),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
        )

        self.feature_fcc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(True),
        )

        self.predict = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b = inputs.size()[0]
        feature_fc = self.feature_fc(inputs.view(b, -1))
        feature_fcc = self.feature_fcc(feature_fc)
        predict = self.predict(feature_fcc)
        return feature_fc, feature_fcc, predict


    def get_output_shape(self):
        return (1024,1,1)


if __name__ == "__main__":

    x = torch.Tensor(3, 10, 5, 5).random_(0, 10)
    # spool = nn.AdaptiveAvgPool2d(1)

    # def cpool():
    #     pool = nn.AdaptiveAvgPool1d(1)
    #     b, c, w, h = x.size()
    #     x = x.view(b, c, w*h).permute(0, 2, 1)
    #     x = pool(x).view(b, 1, w, h)
    #     return x

    # print(cpool(x))
    # print(cpool(x).size())
