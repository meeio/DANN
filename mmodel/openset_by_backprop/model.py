import itertools

import numpy as np
import torch

from .networks.networks import DomainClassifier

from ..basic_module import DAModule, ELoaderIter
from .params import get_params


from mdata.partial.partial_dataset import require_openset_dataloader
from mdata.partial.partial_dataset import OFFICE_HOME_CLASS
from mdata.transfrom import get_transfrom
from mground.gpu_utils import anpai

param = get_params()


def binary_entropy(p):
    p = p.detach()
    e = p * torch.log((p)) + (1 - p) * torch.log((1 - p))
    e = torch.mean(e) * -1
    return e


def norm_entropy(p, reduction="None", all=True):
    p = p.detach()
    n = p.size()[1] - 1
    if not all:
        p = torch.split(p, n, dim=1)[0]
    p = torch.nn.Softmax(dim=1)(p)
    e = p * torch.log((p)) / np.log(n)
    ne = -torch.sum(e, dim=1)
    if reduction == "mean":
        ne = torch.mean(ne)
    return ne


def get_lambda(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def get_lr_scaler(iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75):
    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class OpensetBackprop(DAModule):
    def __init__(self):
        super().__init__(param)

        ## NOTE classes setting adapt from <opensetDa by backprop>

        source_class = set(OFFICE_HOME_CLASS[0:20])
        target_class = set(OFFICE_HOME_CLASS[0:20])

        assert len(source_class.intersection(target_class)) == 20
        assert len(source_class) == 20 and len(target_class) == 20

        class_num = len(source_class) + 1

        self.class_num = class_num
        self.source_class = source_class
        self.target_class = target_class

        self.DECISION_BOUNDARY = self.TARGET.fill_(1)

        self._all_ready()

    def _prepare_data(self):

        back_bone = "alexnet"
        source_ld, target_ld, valid_ld = require_openset_dataloader(
            source_class=self.source_class,
            target_class=self.target_class,
            train_transforms=get_transfrom(back_bone, is_train=True),
            valid_transform=get_transfrom(back_bone, is_train=False),
            params=self.params,
        )

        _, bias_ld, _ = require_openset_dataloader(
            source_class=self.source_class,
            target_class=self.target_class,
            train_transforms=get_transfrom(back_bone, is_train=True),
            valid_transform=get_transfrom(back_bone, is_train=False),
            params=self.params,
        )

        iters = {
            "train": {
                "S": ELoaderIter(source_ld),
                "T": ELoaderIter(target_ld),
                "B": ELoaderIter(bias_ld),
            },
            "valid": ELoaderIter(valid_ld),
        }

        return None, iters

    def _regist_networks(self):

        if True:
            from .networks.alex import AlexNetFc, AlexClassifer

            F = AlexNetFc()
            C = AlexClassifer(
                class_num=self.class_num,
                reversed_coeff=lambda: 1,
            )

        return {"F": F, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.001,
            # "nesterov": True,
            # "lr_mult": {"F": 0.1},
        }

        self.define_loss("global_looss", networks=["C"], optimer=optimer)

        self.define_log("valid_loss", "valid_accu", group="valid")
        self.define_log(
            "classify",
            "adv",
            "e_s",
            "e_t",
            "e_b",
            "ae_s",
            "ae_t",
            "ae_b",
            "ue_s",
            "ue_t",
            "ue_b",
            group="train",
        )

    def _train_step(self, s_img, s_label, t_img, t_label):

        b_img, _ = self.iters["train"]["B"].next()
        b_img = anpai(b_img, use_gpu=True, need_logging=False)

        source_f = self.F(s_img)
        target_f = self.F(t_img)
        bias_f = self.F(b_img)

        s_cls_p, s_un_p = self.C(source_f, adapt=False)
        t_cls_p, t_un_p = self.C(target_f, adapt=True)
        b_cls_p, b_un_p = self.C(bias_f, adapt=True)

        loss_classify = self.ce(s_cls_p, s_label)

        loss_adv = self.bce(t_un_p, self.DECISION_BOUNDARY)

        self._update_logs(
            {
                "classify": loss_classify,
                "adv": loss_adv,
                "e_s": norm_entropy(s_cls_p, reduction="mean", all=False),
                "e_t": norm_entropy(t_cls_p, reduction="mean", all=False),
                "e_b": norm_entropy(b_cls_p, reduction="mean", all=False),
                "ae_s": norm_entropy(s_cls_p, reduction="mean", all=True),
                "ae_t": norm_entropy(t_cls_p, reduction="mean", all=True),
                "ae_b": norm_entropy(b_cls_p, reduction="mean", all=True),
                "ue_s": binary_entropy(s_un_p),
                "ue_t": binary_entropy(t_un_p),
                "ue_b": binary_entropy(b_un_p),
            }
        )
        self._update_loss("global_looss", loss_classify + loss_adv)

        del loss_classify, loss_adv

    def _valid_step(self, img):
        feature = self.F(img)
        prediction, _ = self.C(feature)
        return torch.split(prediction, self.class_num - 1, dim=1)[0]
