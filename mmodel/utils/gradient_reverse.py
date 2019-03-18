from mmodel.basic_module import WeightedModule
from torch.nn import Module
import torch

# def grl_hook(coeff):
#     def fun1(grad):
#         return -coeff*grad.clone()
#     return fun1

class GradReverseLayer(torch.autograd.Function):
    def __init__(self, coeff=lambda: 1):
        super(GradReverseLayer, self).__init__()
        assert callable(coeff) is True
        self.coeff = coeff

    def forward(self, inputs):
        return inputs.view_as(inputs)

    def backward(self, grad_output):
        return -self.coeff() * grad_output.clone() 
