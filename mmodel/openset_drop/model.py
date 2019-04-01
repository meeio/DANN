import itertools

import numpy as np
import torch

from .networks.networks import DomainClassifier

from ..basic_module import DAModule, ELoaderIter
from .params import get_params


from mdata.partial.partial_dataset import require_openset_dataloader
from mdata.partial.partial_dataset import OFFICE_CLASS
from mdata.transfrom import get_transfrom
import numpy as np

param = get_params()


def eval_idx_number(idx, target, number):
    target = target.unsqueeze(1)
    wanted = idx * target.float()
    return torch.sum(wanted == number) / (torch.sum(idx) + 0.001)


def norm_entropy(p, reduction="None"):
    p = p.detach()
    n = p.size()[1] - 1
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


def get_lr_scaler(
    iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75
):
    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class OpensetDrop(DAModule):
    def __init__(self):
        super().__init__(param)

        # self.eval_after = int(0.15 * self.total_steps)
        self.offset = 0.08

        source_class = set(OFFICE_CLASS[0:10])
        target_class = set(OFFICE_CLASS[0:10] + OFFICE_CLASS[20:31])

        assert len(source_class.intersection(target_class)) == 10
        assert len(source_class) == 10 and len(target_class) == 21

        self.source_class = source_class
        self.target_class = target_class
        self.class_num = len(self.source_class) + 1

        self.element_bce = torch.nn.BCELoss(reduction="none")
        self.element_ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.DECISION_BOUNDARY = self.TARGET.fill_(1)

        self._all_ready()

    @property
    def dynamic_offset(self):
        upper = 0.08
        high = 0.078
        low = 0.00
        return upper - get_lambda(
            self.current_step,
            self.total_steps,
            high=high,
            low=low,
            alpha=10,
        )

    def _prepare_data(self):

        back_bone = "alexnet"
        source_ld, target_ld, valid_ld = require_openset_dataloader(
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
            },
            "valid": ELoaderIter(valid_ld),
        }

        return None, iters

    def _regist_networks(self):

        if True:
            from .networks.alex import AlexNetFc, AlexGFC, AlexClassifer

            F = AlexNetFc()
            G = AlexGFC()
            C = AlexClassifer(
                class_num=self.class_num,
                reversed_coeff=lambda: get_lambda(
                    self.current_step, self.total_steps
                ),
            )

        return {"F": F, "G": G, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.001,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "lr_lambda": lambda steps: get_lr_scaler(
                steps, self.total_steps
            ),
            "last_epoch": 0,
        }

        self.define_loss(
            "class_prediction",
            networks=["G", "C"],
            optimer=optimer,
            # decay_op=lr_scheduler,
        )
        self.define_loss(
            "domain_prediction",
            networks=["C"],
            optimer=optimer,
            # decay_op=lr_scheduler,
        )
        self.define_loss(
            "domain_adv",
            networks=["G"],
            optimer=optimer,
            # decay_op=lr_scheduler,
        )

        self.define_log("valid_loss", "valid_accu", group="valid")
        self.define_log(
            "classify",
            "adv",
            "dis",
            "valid_data",
            "outlier_data",
            "drop",
            group="train",
        )

    def _train_step(self, s_img, s_label, t_img, t_label):

        g_source_feature = self.G(self.F(s_img))
        g_target_feature = self.G(self.F(t_img))

        s_predcition, _ = self.C(g_source_feature, adapt=False)
        t_prediction, t_domain = self.C(g_target_feature, adapt=True)

        threshold = (
            norm_entropy(s_predcition, reduction="mean")
            + self.dynamic_offset
        )

        loss_classify = self.ce(s_predcition, s_label)
        ew_dis_loss = self.element_bce(t_domain, self.DECISION_BOUNDARY)

        target_entropy = norm_entropy(t_prediction, reduction="none")
        allowed_idx = (target_entropy < threshold)


        allowed_data_label = torch.masked_select(t_label, mask=allowed_idx)     
        valid = torch.sum(allowed_data_label != self.class_num - 1)
        outlier = torch.sum(
            allowed_data_label == self.class_num - 1
        )
        drop = self.params.batch_size - valid - outlier

        allowed_idx = allowed_idx.float().unsqueeze(1)
        keep_prop = torch.sum(allowed_idx) / self.params.batch_size
        drop_prop = 1 - keep_prop
        dis_loss = torch.mean(ew_dis_loss * (1 - allowed_idx)) * drop_prop
        adv_loss = torch.mean(ew_dis_loss * allowed_idx) * keep_prop

        self._update_logs(
            {
                "classify": loss_classify,
                "dis": dis_loss,
                "adv": adv_loss,
                "valid_data": valid,
                "outlier_data": outlier,
                "drop": drop,
            }
        )

        self._update_losses(
            {
                "class_prediction": loss_classify,
                "domain_prediction": dis_loss + adv_loss,
                "domain_adv": adv_loss / keep_prop,
            }
        )

        del loss_classify, adv_loss

    def _valid_step(self, img):
        feature = self.G(self.F(img))
        prediction, _ = self.C(feature, adapt=False)
        return prediction
