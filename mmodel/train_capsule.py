import logging

from torch import optim as optim
from torch import nn as nn
import torch


class LossBuket(object):
    def __init__(self):
        self.value = None

    def loss(self):
        return self.value


class LossHolder(object):
    def __init__(self):
        self.loss_dic = dict()
        self.fixed = False

    def __getitem__(self, index):
        if self.fixed is True:
            return self.loss_dic[index]

        if not index in self.loss_dic:
            self.loss_dic[index] = LossBuket()
        return self.loss_dic[index]

    def fix_loss_keys(self):
        self.fixed = True


class TrainCapsule(nn.Module):
    """this is a tool class for helping training a network
    """

    # global optimer type and args
    __optim_type__ = None
    __optim_args__ = None

    # global decay op type and args
    __decay_op__ = None
    __decay_args__ = None

    # global decay op function
    # note this function will override decay op
    __recal_lr__ = None

    def __init__(self, optim_loss: LossBuket, optim_networks, tagname=None):
        super(TrainCapsule, self).__init__()

        self.optim_loss = optim_loss
        self.tag = tagname
        self.epoch = 0

        # get all networks, and store them as list
        if not isinstance(optim_networks, tuple):
            networks_list = list()
            networks_list.append(optim_networks)
        else:
            networks_list = optim_networks
        self.optim_network = networks_list

        # get all parameters in network list
        self.all_params = list()
        for i in networks_list:
            self.all_params += list(i.parameters())

        # init optimer base on type and args
        self.optimer = TrainCapsule.__optim_type__(
            self.all_params, **TrainCapsule.__optim_args__
        )

        # init lr_schder for optimer
        if TrainCapsule.__recal_lr__ is None:
            if TrainCapsule.__decay_op__ is not None:
                self.lr_scheduler = TrainCapsule.__decay_op__(
                    self.optimer, **TrainCapsule.__decay_args__
                )

    def __all_networks_call(self, func_name):
        def __one_networkd_call(i):
            func = getattr(i, func_name)
            return func()

        map(__one_networkd_call, self.optim_network)

    def train_step(self):
        self.optimer.zero_grad()
        self.optim_loss.value.backward(retain_graph=True)
        self.optimer.step()

    def registe_default_optimer(optim_type, **kwargs):
        TrainCapsule.__optim_args__ = kwargs
        TrainCapsule.__optim_type__ = optim_type
        logging.info(
            "Resiest %s as default optimizer with args %s ."
            % (optim_type.__name__, kwargs)
        )

    def registe_decay_op(decay_op, **kwargs):
        TrainCapsule.__decay_args__ = kwargs
        TrainCapsule.__decay_op__ = decay_op
        logging.info(
            "Resiest %s as default lr scheduler with args %s ."
            % (decay_op.__name__, kwargs)
        )

    def registe_new_lr_calculator(cal):
        TrainCapsule.__recal_lr__ = cal

    def decary_lr_rate(self):
        self.epoch += 1
        
        if self.__recal_lr__ is not None:
            new_lr = self.__recal_lr__(self.epoch)
            for param_group in self.optimer.param_groups:
                param_group["lr"] = new_lr
            return
        else:
            self.lr_scheduler.step()

        for param_group in self.optimer.param_groups:
            logging.info("Current >learning rate< is >%1.9f< ." % param_group["lr"])
            break


if __name__ == "__main__":

    pass

