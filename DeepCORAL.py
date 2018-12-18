from mmodel.basic_module import DAModule
from mmodel.networks import *
from mmodel.pretrained import *
from mmodel.coral import loss_CORAL

import torch
from params import get_params
from mmodel.mloger import GLOBAL
import logging

import numpy as np


class DeepCORAL(DAModule):

    def __init__(self, params):

        super(DeepCORAL, self).__init__(params)
        self.params = params

        C = Classifer()

        self.C = self.regist_networds(C)

        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD,
            lr=params.lr,
            weight_decay=0.0005,
            momentum=0.9,
            nesterov=True,
        )
        self.TrainCpasule.registe_new_lr_calculator(
            lambda cap, step:
            # params.lr * (1.0 + 0.001 * step) ** (-0.75)
            params.lr
            / ((1.0 + 10.0 * step / self.total_step) ** 0.75)
        )
        self.relr_everytime = True

        # registe loss function
        self.regist_loss("predict", (self.C))


    def train_step(self, s_img, s_label, t_img):

        s_predict = self.C(s_sim)
        t_predict = self.C(t_img)

        l_classifer = self.ce(s_predict, s_label)
        l_coral = loss_CORAL(s_predict, t_predict)

        self.update_loss('predict', l_classifer + l_coral)


    def valid_step(self, img):
        predict = self.C(img)
        return img


if __name__ == "__main__":

    params = get_params()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
    )

    # nadd = MANN(params)
    # nadd.train()

    # from torchvision import models
    # from torchsummary import summary

    # feature = Classifier()
    # feature.weight_init()
    # summary(feature, (3, 32, 32))

