import torch
from torch import nn
from mmodel.basic_module import WeightedModule

class DomainClassifier(WeightedModule):
    def __init__(self, input_dim = 2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,1024)
        self.layer2 = nn.Linear(1024,1024)
        self.layer3 = nn.Linear(1024,1)

        self.layer1.weight.data.normal_(0, 0.01)
        self.layer2.weight.data.normal_(0, 0.01)
        self.layer3.weight.data.normal_(0, 0.3)

        self.layer1.bias.data.fill_(0.0)
        self.layer2.bias.data.fill_(0.0)
        self.layer3.bias.data.fill_(0.0)      

        self.has_init = True

        self.droupout1 = nn.Dropout(0.5)
        self.droupout2 = nn.Dropout(0.5)
        
        self.relu1 = nn.LeakyReLU()  
        self.relu2 = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):
        b = inputs.size()[0]
        x = inputs.view(b,-1)

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.droupout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.droupout2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

                