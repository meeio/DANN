from mmodel.basic_module import DAModule
from mmodel.networks import *
from mmodel.pretrained import *

import torch
from params import get_params
from mmodel.mloger import GLOBAL
import logging


class MANN(DAModule):
    def __init__(self, params):
        super(MANN, self).__init__(params)

        self.params = params

        F = AlexNetFeatureExtrctor()
        B = AlexBottleNeck()
        c, h, w = B.output_shape()
        C = Classifer(params, c * h * w)
        D = DomainClassifer(params, c * h * w)

        self.F, self.B, self.C, self.D = self.regist_networds(F, B, C, D)

        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr, weight_decay=0.0005, momentum=0.9, nesterov=True
        )
        self.TrainCpasule.registe_new_lr_calculator(
            lambda cap, step: params.lr / ( (1 + 10 * step / params.epoch) ** 0.75 )
        )
        # self.relr_everytime = True

        # registe loss function
        self.regist_loss("predict", (self.F, self.B, self.C))
        self.regist_loss("domain", (self.F, self.B, self.D))

    def get_coeff(self):
        sigma=10
        p = self.golbal_step / self.total_step
        llambd = (2.0 / (1.0 + np.exp(-sigma * p))) - 1
        return llambd

    def through(self, img, lable=None):
        feature = self.F(img)
        feature = self.B(feature)

        domain_label = self.TARGET
        predict_loss = None
        if lable is not None:
            domain_label = self.SOURCE
            predict = self.C(feature)
            predict_loss = self.ce(predict, lable)

        domain = self.D(feature, self.get_coeff())
        d_loss = self.bce(domain, domain_label)

        return d_loss, predict_loss


    def train_step(self, s_img, s_label, t_img):

        s_d_loss, s_c_loss = self.through(s_img, s_label)
        t_d_loss, _ = self.through(t_img)

        self.update_loss('domain', (s_d_loss + t_d_loss)/2)
        self.update_loss('predict', s_c_loss)

    def valid_step(self, img):
        feature = self.F(img)
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

