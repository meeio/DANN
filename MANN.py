from mmodel.basic_module import DAModule
from mmodel.networks import *
from mmodel.pretrained import ResNetFeatureExtrctor, AlexNetFeatureExtrctor

import torch
from params import get_params
from mmodel.mloger import GLOBAL
import logging


class MANN(DAModule):
    def __init__(self, params):
        super(MANN, self).__init__(params)

        self.params = params

        F = AlexNetFeatureExtrctor()
        c, h, w = F.output_shape()
        sM = MaskingLayer(c, h * w)
        tM = MaskingLayer(c, h * w)
        D = DomainClassifer(params)

        self.F, self.sM, self.tM, self.D = self.regist_networds(F, sM, tM, D)

        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr, weight_decay=0.0005, momentum=0.9
        )
        self.TrainCpasule.registe_new_lr_calculator(
            lambda cap, step: params.lr / (1 + 10 * step / self.total_step) ** 0.75
        )
        self.relr_everytime = True

        # registe loss function
        self.regist_loss("L_s_d", (self.F, self.sM, self.D))
        self.regist_loss("L_t_d", (self.F, self.tM, self.D))

    def through(self, img, lable=None):

        M = self.tM
        DLabel = self.TARGET
        if lable is not None:
            M = self.sM
            DLabel = self.SOURCE

        feature = self.F(img)

        c_atten, s_atten = M(feature)
        feature = feature * (1 + c_atten)
        feature = feature * (1 + s_atten)

        domain = self.D(feature)
        d_loss = self.bce(domain, DLabel)

        return d_loss

    def train_step(self, s_img, s_label, t_img):

        L_s_d = self.through(s_img, s_label)
        L_t_d = self.through(t_img)

        self.update_loss("L_s_d", L_s_d)
        self.update_loss("L_t_d", L_t_d)

    def valid_step(self, img):
        feature = self.F(img)
        c_atten, s_atten = self.tM(feature)
        feature *= 1 + c_atten
        feature *= 1 + s_atten
        predict = self.D(feature)
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
    # alex = models.alexnet(pretrained=True)
    # layers = list(alex.children())[0][:-1]

    # print(layers)
    

