from mmodel.basic_module import WeightedModule
from torch import nn
import numpy as np

class FeatureExtractor(WeightedModule):
    def __init__(self, param):
        super().__init__()

        nf = param.nf
        # input feature with shape [32,32,3]
        # 32 + 4 - 5 / 2 + 1
        self.feature_conv = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=5, =2, padding=2),
            nn.BatchNorm2d(nf),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            # [16,16]
            nn.Conv2d(64, 2 * nf, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2 * nf),
            nn.Dropout2d(0.3),
            nn.ReLU(True),
            # [8,8]
            nn.Conv2d(128, 4 * nf, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4 * nf),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            # [4,4]
        )

        self.feature_full = nn.Sequential(
            nn.Linear(4 * nf * 4 * 4, 8 * nf),
            nn.BatchNorm1d(8 * nf),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        conv = self.feature_conv(x)
        conv = conv.view(conv.size(0), -1)
        feature = self.feature_full(conv)
        return feature

class Classifier(WeightedModule):

    def __init__(self, param):
        super().__init__()

        nf = param.nf
        class_num = param.class_num
        self.predict = nn.Sequential(
            # nn.Linear(8 * nf, 24 * nf),
            # nn.ReLU(True),

            # nn.Linear(24 * nf, 16 * nf),
            # nn.ReLU(True),

            nn.Linear(8 * nf, class_num),
        )
        
    def forward(self, x):
        predict = self.predict(x)
        return predict       

class DomainClassifer(WeightedModule):
    def __init__(self, param):
        super().__init__()
    
        nf = param.nf
        hidden_size = nf * 8
        self.iter_num = 0
        self.predict = nn.Sequential(
            nn.Linear(nf*8, hidden_size),
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

    def forward(self, x, coeff):
        x = x * 1.0
        x.register_hook(lambda grad: grad * -1 * coeff)
        domain = self.predict(x)
        return domain
    
