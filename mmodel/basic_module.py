import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import torchvision
import numpy as np

from mground.gpu_utils import anpai

import os
from mmodel.train_capsule import TrainCapsule, LossHolder
from mmodel.mloger import LogCapsule

import mdata.dataloader as mdl
import mdata.dataloader as mdl
from abc import ABC, abstractclassmethod


def _basic_weights_init_helper(modul, params=None):
    """give a module, init it's weight
    
    Args:
        modul (nn.Module): torch module
        params (dict, optional): Defaults to None. not used.
    """
    for m in modul.children():
        # init Conv2d with norm
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            init.constant_(m.bias, 0)
        # init BatchNorm with norm and constant
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)
        # init full connect norm
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            _basic_weights_init_helper(m)

        if isinstance(m, WeightedModule):
            m.has_init = True


class WeightedModule(nn.Module):

    static_weight_handler = None

    def __init__(self):
        super().__init__()
        self.has_init = False
        self.lr_mult = 1

    def __call__(self, *input, **kwargs):
        if not self.has_init:
            raise Exception("Init weight before you use it")
        return super().__call__(*input, **kwargs)

    def weight_init(self, handler=None, record_path=None):
        """initial weights with help function `handler` or a path of check point `record_path` to restore, if neither of those are provide, leave init jod to torch inner function.

            handler (function, optional): Defaults to None. use to init weights, such function must has sigture of `handler(Module)`.
            record_path (str, optional): Defaults to None. use to locate cheack point file.
        """

        """ init weights with torch inner function 
        """

        name = self.__class__.__name__
        str = name + "'s weights init from %s."

        if self.has_init:
            logging.info(str % "Pretraied Model")
            return

        if record_path is not None:
            f = torch.load(record_path)
            self.load_state_dict(f)
            logging.info(str % "check point")

        elif handler is not None:
            handler(self)
            logging.info(str % "provided init function")

        elif WeightedModule.static_weight_handler is not None:
            WeightedModule.static_weight_handler(self)
            logging.info(str % "WeightedModule's register init function")

        else:
            logging.info(str % "Pytorch's inner mechainist")

        self.has_init = True

    def register_weight_handler(handler):
        WeightedModule.static_weight_handler = handler

    def save_model(self, path, tag):
        f = path + tag
        torch.save(self.cpu().state_dict(), f)
        return os.path.abspath(f)


class DAModule(ABC, nn.Module):
    def __init__(self, params):
        super().__init__()

        # set global dict
        self.losses = LossHolder()
        self.train_caps = dict()
        self.loggers = dict()
        self.params = params
        self.networks = None
        self.golbal_step = 0
        self.current_epoch = 0
        self.relr_everytime = False
        self.best_accurace = 0.0

        # set usefully loss function
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

        # set default weight initer
        WeightedModule.register_weight_handler(_basic_weights_init_helper)

        self.TrainCpasule = TrainCapsule
        # set default optim function
        TrainCapsule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr, weight_decay=0.0005, momentum=0.9
        )

        # define valid losses
        self.log_valid_loss = LogCapsule(
            self.losses["valid_loss"], "valid_loss", to_file=params.log
        )
        self.log_valid_acrr = LogCapsule(
            self.losses["valid_acuu"], "valid_acuu", to_file=params.log
        )

        # generate source and target data set for trian and test
        # s_set_name = getattr(mdl.DSNames, params.sdsname)
        # t_set_name = getattr(mdl.DSNames, params.tdsname)

        self.t_s_data_set, self.t_s_data_loader = mdl.load_img_dataset(
            "OfficeHome", "Ar", params.batch_size
        )
        self.t_t_data_set, self.t_t_data_loader = mdl.load_img_dataset(
            "OfficeHome", "Pr", params.batch_size
        )
        self.v_t_data_set, self.v_t_data_loader = mdl.load_img_dataset(
            "OfficeHome", "Pr", params.batch_size, test=True
        )

        # set total train step
        self.total_step = int(
            int(min(len(self.t_s_data_set), len(self.t_t_data_set)) / params.batch_size)
            * params.epoch
        )

        # init global label
        self.TARGET, self.SOURCE = self.__batch_domain_label__(params.batch_size)

    def regist_loss(self, loss_name, networks):
        t = TrainCapsule(self.losses[loss_name], networks)
        l = LogCapsule(self.losses[loss_name], loss_name, to_file=self.params.log)
        self.train_caps[loss_name] = t
        self.loggers[loss_name] = l

    def regist_networds(self, *networks):
        for i in networks:
            i.weight_init()
        networks = anpai(networks, self.params.use_gpu)
        self.networks = networks
        return networks

    def valid(self, step=None):

        params = self.params

        # set all networks to eval mode
        for i in self.networks:
            i.eval()

        # def a helpler fucntion
        def valid_a_set(data_loader):

            for _, (img, label) in enumerate(data_loader):

                if len(img) != self.params.batch_size:
                    continue

                img, label = anpai((img, label), params.use_gpu, need_logging=False)

                # get result from a valid_step
                predict = self.valid_step(img)

                # calculate valid loss and make record
                self.losses["valid_loss"].value = self.ce(predict, label)
                self.log_valid_loss.record()

                # calculate valid accurace and make record
                current_size = label.size()[0]
                _, predic_class = torch.max(predict, 1)
                corrent_count = (predic_class == label).sum().float()
                self.losses["valid_acuu"].value = corrent_count / current_size
                self.log_valid_acrr.record()

            self.log_valid_loss.log_current_avg_loss(self.golbal_step)
            accu = self.log_valid_acrr.log_current_avg_loss(self.golbal_step)

            return accu

        # valid on target data
        accu = valid_a_set(self.t_t_data_loader)
        return accu

    def train(self):

        # fix loss key to prenvent missing
        self.losses.fix_loss_keys()

        # calculate record per step
        total_datas = min((len(self.t_s_data_set), len(self.t_t_data_set)))
        record_per_step = total_datas / (
            self.params.batch_size * self.params.log_per_epoch
        )
        record_per_step = int(record_per_step)

        for epoch in range(self.params.epoch):
            # set all networks to train mode
            for i in self.networks:
                i.train(True)

            if not self.relr_everytime:
                for c in self.train_caps.values():
                    c.decary_lr_rate()

            # begain a epoch
            for epoch_step, (sorce, target) in enumerate(
                zip(self.t_s_data_loader, self.t_t_data_loader)
            ):
                # send train data to wantted device
                s_img, s_label = sorce
                t_img, _ = target

                if len(s_img) != len(t_img):
                    continue

                s_img, s_label, t_img = anpai(
                    (s_img, s_label.long(), t_img),
                    self.params.use_gpu,
                    need_logging=False,
                )

                if self.relr_everytime:
                    for c in self.train_caps.values():
                        c.decary_lr_rate()

                # begain a train step
                self.train_step(s_img, s_label, t_img)

                # make record if need
                self.golbal_step += 1
                if self.golbal_step % record_per_step == (record_per_step - 1):
                    for v in self.loggers.values():
                        v.log_current_avg_loss(self.golbal_step)

            # after an epoch begain valid
            if (self.current_epoch % 5) == 4:
                accu = self.valid()
                self.best_accurace = max((self.best_accurace, accu))

            logging.info(
                "Epoch %3d ends. \t Remain %3d epoch to go. "
                % (self.current_epoch + 1, self.params.epoch - self.current_epoch -1)
            )

            logging.info(
                "Current best accurace is %3.2f ." % (self.best_accurace * 100)
            )
            self.current_epoch += 1

    @abstractclassmethod
    def train_step(self, s_img, s_label, t_img):
        pass

    @abstractclassmethod
    def valid_step(self, img):
        pass

    def update_loss(self, loss_name, value):
        self.losses[loss_name].value = value
        self.loggers[loss_name].record()
        self.train_caps[loss_name].train_step()

    def __batch_domain_label__(self, batch_size):
        # Generate all Source and Domain label.
        SOURCE = 1
        TARGET = 0

        sd = torch.Tensor(batch_size, 1).fill_(SOURCE)
        td = torch.Tensor(batch_size, 1).fill_(TARGET)

        sd, td, = anpai((sd, td), self.params.use_gpu, False)
        return sd, td
