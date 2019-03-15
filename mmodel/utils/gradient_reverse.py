from mmodel.basic_module import WeightedModule
from torch.nn import Module
import torch

class GradReverseLayer(Module):
    def __init__(self, coeff=lambda: 1):
        super(GradReverseLayer, self).__init__()
        assert callable(coeff) is True
        self.has_init = True
        self.coeff = coeff
        # self.trash = torch.nn.Linear(1,1)

    def forward(self, inputs):
        return inputs.view_as(inputs)

    def backward(self, grad_output):
        return grad_output * -self.coeff()