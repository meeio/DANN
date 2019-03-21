import torch

from mmodel import get_module


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    name = input('model name:')
    # name = 'open'
    _, A = get_module(name)
    A.train_module()
