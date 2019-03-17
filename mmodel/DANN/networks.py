import torch
from torch import nn
from mmodel.basic_module import WeightedModule

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class DomainClassifier(WeightedModule):
    def __init__(self, input_dim = 2048, reversed_function=None):
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

        self.droupout1 = nn.Dropout(0.5)
        self.droupout2 = nn.Dropout(0.5)

        self.has_init = True

        self.relu1 = nn.LeakyReLU(inplace=True)  
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # assert callable(reversed_hook)
        self.reversed_function = reversed_function


    def forward(self, inputs):
        inputs = self.reversed_function(inputs)
        
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


class BottleneckedClassifier(WeightedModule):

    def __init__(self, input_dim, class_num, bottleneck_dim = 256):
        super(BottleneckedClassifier, self).__init__()   

        bottleneck = nn.Linear(input_dim, bottleneck_dim)
        classifer = nn.Linear(bottleneck_dim, class_num)

        nn.init.xavier_normal_(bottleneck.weight)
        nn.init.constant_(bottleneck.bias, 0)

        nn.init.xavier_normal_(classifer.weight)
        nn.init.constant_(classifer.bias, 0)

        self.bottleneck = bottleneck
        self.classifer = classifer

        self.has_init = True
    
    def forward(self, inputs):

        feature = self.bottleneck(inputs)
        prediction = self.classifer(feature)

        return feature, prediction
