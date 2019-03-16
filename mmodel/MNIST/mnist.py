import itertools

from mdata.partial_folder import MultiFolderDataHandler
from mground.gpu_utils import current_gpu_usage
from mground.math_utils import entropy, make_weighted_sum
from mtrain.mloger import GLOBAL, LogCapsule

import numpy as np
from ..basic_module import DAModule
from ..utils.thuml_feature_extractor import AlexNetFeatureExtractor
from ..utils.gradient_reverse import GradReverseLayer
from .network import *
from .params import get_params


param = get_params()
param.dataset = None
param.source = 'MNIST'
param.target = 'MNIST'


def get_lambda(iter_num, max_iter=10000.0, high=1.0, low=0.0, alpha=10.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def get_lr_scaler(iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75):

    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class MNIST(DAModule):
    def __init__(self):
        super(MNIST, self).__init__(param)
        self._all_ready()

    def _regist_networks(self):
        C = DTN()
        return {"C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": self.params.lr,
            "momentum": 0.95,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "lr_lambda": lambda steps: get_lr_scaler(steps, self.total_steps),
            "last_epoch": 0,
        }

        self.define_loss(
            "global_looss",
            networks=["C"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self.define_log("classify")

    def _train_step(self, s_img, s_label, t_img):

        predict = self.C(s_img)

        loss_classify = self.ce(predict, s_label)


        self._update_logs({"classify": loss_classify})
        self._update_loss("global_looss", loss_classify)

        del loss_classify

    def _valid_step(self, img):
        predict = self.C(img)
        return predict