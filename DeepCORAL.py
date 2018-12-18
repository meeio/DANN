from mmodel.basic_module import DAModule
from mmodel.networks import *
from mmodel.pretrained import *
from mmodel.coral import loss_CORAL


import torch
from params import get_params
from mmodel.mloger import GLOBAL
import logging

from mground.plot_utils import plot_all
from mmodel.mloger import read_step_and_loss


import numpy as np


class DeepCORAL(DAModule):

    def __init__(self, params):

        super(DeepCORAL, self).__init__(params)
        self.params = params

        C = Classifier(params)
        self.C = self.regist_networds(C)

        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.Adam,
            lr=params.lr,
        )

        # registe loss function
        self.regist_loss("predict", self.C)


    def train_step(self, s_img, s_label, t_img):

        s_feature, s_predict = self.C(s_img)
        t_feature, t_predict = self.C(t_img)

        l_classifer = self.ce(s_predict, s_label)
        l_coral = loss_CORAL(s_feature, t_feature)

        self.update_loss('predict', l_classifer + self.params.coral_param * l_coral)


    def valid_step(self, img):
        _, predict = self.C(img)
        return predict


if __name__ == "__main__":

    params = get_params()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
    )

    coral = DeepCORAL(params)
    coral.train()

    # record_dat = read_step_and_loss(
    #     train_loss = r'G:\VS Code\DANN\_MLOGS\CORAL1\predict.log',
    #     valid_loss = r'G:\VS Code\DANN\_MLOGS\CORAL1\valid_loss.log',
    #     valid_accur = r'G:\VS Code\DANN\_MLOGS\CORAL1\valid_acuu.log',
    # )

    # plot_all(record_dat, tagname='with CORAL')

    # from torchvision import models
    # from torchsummary import summary

    # feature = Classifier()
    # feature.weight_init()
    # summary(feature, (3, 32, 32))

