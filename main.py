import torch

from mmodel import get_module
from mtrain.watcher import watcher
import sys


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    # name = input('model name:')
    name = "OPENDP"
    try:
        _, A = get_module(name)
        from mmodel.basic_params import basic_params, parser

        if basic_params.make_record:
            watcher.prepare_notes(name, basic_params.tag)
        A.train_module()

    finally:
        watcher.to_json()
