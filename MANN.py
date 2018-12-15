from mmodel.basic_module import DAModule
from mmodel.networks import *
from mmodel.pretrained import *

import torch
from params import get_params
from mmodel.mloger import GLOBAL
import logging

import numpy as np


class MANN(DAModule):
    def __init__(self, params):
        super(MANN, self).__init__(params)

        self.params = params

        F1 = AlexNetFeatureExtrctor()
        F2 = AlexBottleNeck()
        c, h, w = F2.output_shape()

        B = BottleNeck(params, c * h * w)
        c1, h1, w1 = B.output_shape()

        C = Classifer(params, c1 * h1 * w1)
        D = DomainClassifer(params, c1 * h1 * w1)

        self.F1, self.F2, self.B, self.C, self.D = self.regist_networds(F1, F2, B, C, D)

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
        self.regist_loss("predict", (self.B, self.C))
        self.regist_loss("domain", (self.F1, self.F2, self.B, self.D))

    def get_coeff(self, high=1):
        sigma = 10
        p = self.golbal_step / (self.total_step)
        llambd = np.float((2.0 * high / (1.0 + np.exp(-sigma * p))) - high)
        return llambd

    def through(self, img, lable=None):
        feature = self.F1(img)
        feature = self.F2(feature)
        feature = self.B(feature)

        domain_label = self.TARGET
        predict_loss = None
        if lable is not None:
            domain_label = self.SOURCE
            predict = self.C(feature)
            predict_loss = nn.CrossEntropyLoss()(predict, lable)

        domain = self.D(feature, self.get_coeff())
        d_loss = nn.BCELoss()(domain, domain_label)

        return d_loss, predict_loss

    def train_step(self, s_img, s_label, t_img):

        s_d_loss, s_c_loss = self.through(s_img, s_label)
        t_d_loss, _ = self.through(t_img)

        self.update_loss("domain", s_d_loss/2 + t_d_loss/2)
        self.update_loss("predict", s_c_loss)


    def valid_step(self, img):
        feature = self.F1(img)
        feature = self.F2(feature)
        feature = self.B(feature)
        predict = self.C(feature)

        return predict


if __name__ == "__main__":

    params = get_params()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
    )

    nadd = MANN(params)
    nadd.train()

    # from torchvision import models
    # from torchsummary import summary

    # alex = models.alexnet(pretrained=True)
    # alex = AlexNetFeatureExtrctor()
    # alex = AlexNetClassifier(params)
    # summary(alex, (256, 6, 6), 64)
    # print(alex)

