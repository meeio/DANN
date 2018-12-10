from mmodel.basic_module import DAModule
from mmodel.networks import *

import torch
from params import get_params
from mmodel.mloger import GLOBAL
import logging


class NADD(DAModule):
    def __init__(self, params):
        super().__ini__(params)

        f = FeatureExtractor(params)
        c = Classifier(params)
        d = DomainClassifer(params)

        self.F, self.C, self.D = self.regist_networds(f, c, d)

        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr, weight_decay=0.0005, momentum=0.9
        )

        self.TrainCpasule.registe_new_lr_calculator(
            lambda cap, epoch: params.lr / (1 + 10 * epoch / params.epoch) ** 0.75,
        )

        self.regist_loss("predict", (self.F, self.C))
        self.regist_loss("dis", (self.F, self.D))


    def get_coeff(self, sigma=10):
        p = self.golbal_step / self.total_step
        llambd = (2 / (1 + np.exp(-sigma * p))) - 1
        return llambd

    def train_step(self, s_img, s_label, t_img):

        s_feature = self.F(s_img)
        t_feature = self.F(t_img)

        s_domain = self.D(s_feature, self.get_coeff())
        t_domain = self.D(t_feature, self.get_coeff())

        predict = self.C(s_feature)

        dis_loss = self.bce(s_domain, self.SOURCE) + self.bce(t_domain, self.TARGET)
        self.update_loss("dis", dis_loss)

        predict_loss = self.ce(predict, s_label)
        self.update_loss("predict", predict_loss)

    def valid_step(self, img):
        feature = self.F(img)
        predict = self.C(feature)
        return predict


if __name__ == "__main__":

    params = get_params()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, 
        format=" \t | %(levelname)s |==> %(message)s",
    )

    nadd = NADD(params)
    nadd.train()
