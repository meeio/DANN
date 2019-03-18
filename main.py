import torch

from mmodel import get_module


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    name = input('model name:')
    _, A = get_module(name)
    A.train_module()
