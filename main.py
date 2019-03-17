from mmodel import get_module
import torch


torch.cuda.empty_cache()
_, A = get_module("DANN")
A.train_module()

