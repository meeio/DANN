import itertools

import torch

from mdata.partial_folder import MultiFolderDataHandler
from mground.gpu_utils import current_gpu_usage
from mground.math_utils import entropy, make_weighted_sum
from mtrain.mloger import GLOBAL, LogCapsule

import numpy as np
from ..basic_module import DAModule
from ..utils.thuml_feature_extractor import AlexNetFc
from ..utils.gradient_reverse import GradReverseLayer
from .networks import *
from .params import get_params


param = get_params()


def get_lambda(iter_num, max_iter=10000.0, high=1.0, low=0.0, alpha=10.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def get_lr(iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75):
    lr = np.float(
        init_lr * (1 + alpha * (iter_num / max_iter)) ** (-power)
    )
    return lr


class DANN(DAModule):
    def __init__(self):
        super(DANN, self).__init__(param)
        self._all_ready()

    def _regist_networks(self):
        def grader_reverse_lambda():
            return get_lambda(self.current_step, max_iter=self.total_steps)

        F = AlexNetFc(bottleneck_dim=256, new_cls=True, class_num=31)
        D = nn.Sequential(
            GradReverseLayer(coeff=grader_reverse_lambda),
            DomainClassifier(input_dim=256),
        )

        return {"F": F, "D": D}

    def _regist_losses(self):

        ## WARM need weight decay?

        optimer = {
            "type": torch.optim.SGD,
            "lr": self.params.lr,
            "momentum": 0.95,
            "lr_mult": {"F": 0.1},
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "lr_lambda": lambda steps: get_lr(steps, self.total_steps),
        }

        self.define_loss(
            "loss_prediction",
            networks=["F"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )
        self.define_loss(
            "loss_discrim",
            networks=["D", "F"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self.define_log("classify", "discrim")

    def _train_step(self, s_img, s_label, t_img):
        def for_losses(img, label):

            feature, predict_class = self.F(img)
            predict_domain = self.D(feature)

            if label is None:
                domain = self.TARGET
                loss_classifi = 0
            else:
                domain = self.SOURCE
                loss_classifi = self.ce(predict_class, label)

            loss_discrimi = self.bce(predict_domain, domain)

            return loss_classifi, loss_discrimi

        _, lt_dis = for_losses(t_img, None)
        ls_class, ls_dis = for_losses(s_img, s_label)

        loss_classify = ls_class
        loss_dis = (ls_dis + lt_dis) 

        self._update_logs({"classify": loss_classify, "discrim": loss_dis})

        self._update_losses(
            {"loss_prediction": loss_classify, 'loss_discrim': loss_dis}
        )

    def _valid_step(self, img):
        _, predict = self.F(img)
        return predict
