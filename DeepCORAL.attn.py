from mmodel.basic_module import DAModule
from mmodel.CORAL.attnCORAL.networks import *
from mmodel.CORAL.coral import loss_CORAL


import torch
from params import get_param_parser
from mmodel.mloger import GLOBAL
import logging

def get_coral_param():
    parser = get_param_parser()

    parser.add_argument(
        "--coral_param", type=float, default=4, help="coral parameter."
    )

    parser.add_argument(
        "--normF_params", type=float, default=0.1, help="coral parameter."
    )

    return parser.parse_args()

class AttenCORAL(DAModule):

    def __init__(self, params):

        super(AttenCORAL, self).__init__(params)
        self.params = params

        F = FeatureExtractor(params)
        C = Classifier()
        D = DomainClassifier()

        self.F, self.C, self.D = self.regist_networds(F, C, D)

        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.Adam,
            lr=params.lr,
        )

        # registe loss function
        self.regist_loss("predict", (self.F, self.C))
        # self.regist_loss("coral", (self.F, self.D))
        self.regist_loss("domain", self.D)
        self.regist_loss("F_norm", (self.D, self.C))

    def through(self, img, lable=None):
        feature = self.F(img)
        c_feature, predict = self.C(feature)
        d_feature_1, d_feature_2, domain = self.D(feature)

        domain_label = self.TARGET
        predict_loss = None
        if lable is not None:
            domain_label = self.SOURCE
            predict_loss = nn.CrossEntropyLoss()(predict, lable)

        d_loss = self.bce(domain, domain_label)

        F_norm_loss = torch.mean(
            torch.mul(
                (c_feature - d_feature_1), (c_feature - d_feature_1)
            )
        )

        return d_loss, predict_loss, F_norm_loss, d_feature_2


    def train_step(self, s_img, s_label, t_img):

        s_d, s_p, s_F, s_f = self.through(s_img, s_label)
        t_d, _, t_F, t_f = self.through(t_img)

        l_coral = loss_CORAL(s_f, t_f)

        # self.update_loss('coral', self.params.coral_param * l_coral)
        self.update_loss('predict', s_p)
        self.update_loss('domain', s_d + t_d)
        # self.update_loss('F_norm', self.params.normF_params * t_F)

    def valid_step(self, img):
        feature = self.F(img)
        _, predict = self.C(feature)
        return predict


if __name__ == "__main__":

    params = get_coral_param()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, 
        format=" \t | %(levelname)s |==> %(message)s"
    )

    if True:
        coral = AttenCORAL(params)
        coral.train()

    # from torchvision import models
    # from torchsummary import summary

    # feature = Classifier(params)
    # feature.weight_init()
    # feature = feature.cuda()
    # summary(feature, (1, 32, 32))

