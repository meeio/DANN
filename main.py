from mmodel import get_module
import logging

logging.basicConfig(
    level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
)

_, A = get_module('MSDA')
A.train_module()
