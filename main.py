import torch

from mmodel import get_module

torch.cuda.empty_cache()
_, A = get_module("DANN")
A.train_module()
