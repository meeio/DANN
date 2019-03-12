import torch
import torch.nn as nn
import torch.nn.init as init
import logging

from mground.gpu_utils import anpai

import os
from mtrain.train_capsule import TrainCapsule, LossHolder
from mtrain.mloger import LogCapsule

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
        # if not self.has_init:
        #     raise Exception("Init weight before you use it")
        return super().__call__(*input, **kwargs)

    def weight_init(self, handler=None, record_path=None):
        """initial weights with help function `handler` or a path of check point `record_path` to restore, if neither of those are provide, leave init 
        jod to torch inner function.

            handler (function, optional): Defaults to None. use to init weights, such function must has sigture of `handler(Module)`.
            record_path (str, optional): Defaults to None. use to locate cheack point file.
        """

        """ init weights with torch inner function 
        """

        name = self.__class__.__name__
        str = name + "'s weights init from %s."

        if self.has_init:
            # logging.info(str % "Pretraied Model")
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


class ELoaderIter:

    """ A helper class which iter a dataloader endnessly
    
    Arguments:
        dataloader {DataLoader} -- Dataloader want to iter
    
    """

    def __init__(self, dataloader):
        self.l = dataloader
        self.it = None

    def next(self, need_end=False):
        """ return next item of dataloader, if use 'endness' mode, 
        the iteration will not stop after one epoch
        
        Keyword Arguments:
            need_end {bool} -- weather need to stop after one epoch
             (default: {False})
        
        Returns:
            list -- data
        """

        if self.it == None:
            self.it = iter(self.l)

        try:
            i = next(self.it)
        except Exception:
            self.it = iter(self.l)
            i = next(self.it) if not need_end else None

        return i


class TrainableModule(ABC):

    """ An ABC, a Tranable Module is a teample class need to define
    data process, train step and eval_step

    """

    def __init__(self, params):

        super(TrainableModule, self).__init__()

        self.params = params
        self.log_step = self.params.log_per_step
        self.eval_step = self.params.eval_per_step
        self.TrainCpasule = TrainCapsule

        self.relr_everytime = False
        self.eval_once = False

        self.steps = self.params.steps
        self.golbal_step = 0.0
        self.current_epoch = 0.0

        self.losses = LossHolder()
        self.train_caps = dict()
        self.loggers = dict()

        T = torch.zeros(1)
        S = torch.ones(1)
        self.T, self.S = anpai(
            (T, S), use_gpu=params.use_gpu, need_logging=False
        )

    def _all_ready(self):

        # Registe all needed work
        # registe weights initial funcion
        WeightedModule.register_weight_handler(_basic_weights_init_helper)
        # get all networks and init weights
        networks = self._regist_networks()
        assert type(networks) is dict

        for _, i in networks.items():
            try:
                i.weight_init()
            except Exception:
                pass
        # send networks to gup
        networks = {
            i: anpai(j, use_gpu=self.params.use_gpu)
            for i, j in networks.items()
        }
        # make network be class attrs
        for i, j in networks.items():
            self.__setattr__(i, j)
        self.networks = networks

        # regist losses
        # train_caps used to update networks
        # loggers used to make logs
        self._regist_losses()

        # generate train dataloaders and valid dataloaders
        # data_info is a dict contains basic data infomations
        d_info, iters = self._prepare_data()
        self.iters = iters

    @abstractclassmethod
    def _prepare_data(self):
        """ handle dataset to produce dataloader
        
        Returns:
            list -- a dict of datainfo, a list of train dataloader
            and a list of valid dataloader.
        """

        data_info = dict()
        train_loaders = list()
        valid_loaders = list()
        return data_info, train_loaders, valid_loaders

    @abstractclassmethod
    def _feed_data(self, mode):
        """ feed example based on dataloaders

        Returns:
            list -- all datas needed.
        """
        datas = list()
        return datas

    @abstractclassmethod
    def _regist_losses(self):
        """ regist lossed with the help of regist loss

        Returns:
            list -- all datas needed.
        """
        return

    @abstractclassmethod
    def _regist_networks(self):
        """ feed example based on dataloaders

        Returns:
            list -- all datas needed.
        """
        networks = dict()
        return networks

    @abstractclassmethod
    def _train_process(self, datas, **kwargs):
        """process to train 
        """
        pass

    @abstractclassmethod
    def _eval_process(self, datas, **kwargs):
        """process to eval 
        """
        pass

    @abstractclassmethod
    def _log_process(self):
        """ make logs
        """
        return

    def regist_loss(self, loss_name, *networks_key):
        """registe loss according loss_name and relative networks.
        after this process the loss will bind to the weight of networks, which means this loss will used to update weights of provied networks.
        
        Arguments:
            loss_name {str} -- loss name
            networks_key {list} -- relative network names
        """

        if type(networks_key[0]) is str:
            networks = [self.networks[i] for i in networks_key]
        else:
            networks = networks_key

        self.losses[loss_name]
        t = TrainCapsule(self.losses[loss_name], networks)
        self.train_caps[loss_name] = t

    def regist_log(self, loss_name):
        self.losses[loss_name]
        l = LogCapsule(
            self.losses[loss_name], loss_name, to_file=self.params.log
        )
        self.loggers[loss_name] = l

    def train_module(self, **kwargs):

        # fixed loss key to prenvent missing
        self.losses.fix_loss_keys()

        # set all networks to train mode
        for _, i in self.networks.items():
            i.train(True)

        for _ in range(self.steps):

            datas = self._feed_data(mode="train")

            datas = anpai(datas, self.params.use_gpu, need_logging=False)
            self._train_process(datas, **kwargs)

            # re calculate learning rates
            if self.relr_everytime:
                for c in self.train_caps.values():
                    c.decary_lr_rate()

            # making log
            if self.golbal_step % self.log_step == (self.log_step - 1):
                self._log_process()

            # begain eval
            if self.golbal_step % self.eval_step == (self.eval_step - 1):
                self.eval_module(**kwargs)
                # set all networks to train mode
                for _, i in self.networks.items():
                    i.train(True)

            self._finish_a_train_process()

    def _finish_a_train_process(self):
        self.golbal_step += 1

    def eval_module(self, **kwargs):
        # set all networks to eval mode
        for _, i in self.networks.items():
            i.eval()

        while True:
            datas = self._feed_data(mode="valid")

            if datas is not None:
                datas = anpai(
                    datas, self.params.use_gpu, need_logging=False
                )
            self._eval_process(datas, **kwargs)

            if datas is None or self.eval_once:
                break

    def _update_loss(self, loss_name, value, retain_graph=True):
        self.losses[loss_name].value = value
        self.train_caps[loss_name].train_step(retain_graph)

    def _update_log(self, loss_name, value):
        self.losses[loss_name].value = value
        self.loggers[loss_name].record()


class DAModule(TrainableModule):
    def __init__(self, params):
        super(DAModule, self).__init__(params)

        self.relr_everytime = False
        self.best_accurace = 0.0
        self.total = self.corret = 0

        # set usefully loss function
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

        self.TrainCpasule = TrainCapsule
        # set default optim function
        TrainCapsule.registe_default_optimer(
            torch.optim.SGD,
            lr=params.lr,
            weight_decay=0.0005,
            momentum=0.9,
        )

        # define valid losses
        self.log_valid_loss = LogCapsule(
            self.losses["valid_loss"], "valid_loss", to_file=params.log
        )
        self.log_valid_acrr = LogCapsule(
            self.losses["valid_acuu"], "valid_acuu", to_file=params.log
        )

        # set total train step
        self.total_step = self.params.steps

        # init global label
        self.TARGET, self.SOURCE = self.__batch_domain_label__(
            params.batch_size
        )

    @abstractclassmethod
    def _train_step(self, s_img, s_label, t_img):
        pass

    @abstractclassmethod
    def _valid_step(self, img):
        pass

    def _prepare_data(self):

        params = self.params

        dataset = params.dataset
        source = params.sdsname
        target = params.tdsname

        def get_set(dataset, domain, split):
            if dataset is None:
                dataset = mdl.get_dataset(
                    dataset=domain, domain=None, split=split
                )
            else:
                dataset = mdl.get_dataset(
                    dataset=dataset, domain=domain, split=split
                )
            return dataset

        train_S_set = mdl.get_dataset(dataset, source, split="train")
        train_T_set = mdl.get_dataset(dataset, target, split="train")
        valid_set = mdl.get_dataset(dataset, target, split="test")

        def get_loader(dataset, shuffle):
            DataLoader = torch.utils.data.DataLoader
            train_S_loader = DataLoader(
                train_S_set,
                batch_size=params.batch_size,
                drop_last=False,
                shuffle=shuffle,
            )

        train_S_l = get_loader(train_S_set, shuffle=False)
        train_T_l = get_loader(train_T_set, shuffle=True)
        valid_l = get_loader(valid_set, shuffle=False)

        return None, (train_S_l, train_T_l), (valid_l,)

    def _feed_data(self, mode, *args, **kwargs):

        if self.iters[mode] is None:
            if mode == "train":
                sloader, tloader = self.train_loaders
                eits = ELoaderIter(sloader), ELoaderIter(tloader)
            else:
                loader = self.valid_loaders[0]
                eits = ELoaderIter(loader)
            self.iters[mode] = eits

        its = self.iters[mode]
        if mode == "train":
            s_img, s_label = its[0].next()
            t_img, _ = its[1].next()
            return s_img, s_label, t_img
        else:
            return its.next(need_end=True)

    def _train_process(self, datas):

        s_img, s_label, t_img = datas

        # re calculate learning rates
        if self.relr_everytime:
            for c in self.train_caps.values():
                c.decary_lr_rate()

        # begain a train step
        self._train_step(s_img, s_label, t_img)

    def _log_process(self):

        logging.info(
            "Steps %3d ends. Remain %3d steps to go. Fished %.2f%%"
            % (
                self.golbal_step + 1,
                self.params.steps - self.golbal_step - 1,
                (self.golbal_step + 1) / (self.params.steps + 1) * 100,
            )
        )

        logging.info(
            "Current best accurace is %3.3f%%."
            % (self.best_accurace * 100)
        )

        for v in self.loggers.values():
            v.log_current_avg_loss(self.golbal_step + 1)

    def _eval_process(self, datas):

        params = self.params

        end_epoch = datas is None

        def handle_datas(datas):

            right = total = 0

            img, label = datas
            # get result from a valid_step
            predict = self._valid_step(img)

            self.losses["valid_loss"].value = self.ce(predict, label)
            self.log_valid_loss.record()

            # calculate valid accurace and make record
            current_size = label.size()[0]
            _, predic_class = torch.max(predict, 1)
            corrent_count = (predic_class == label).sum().float()

            self.losses["valid_acuu"].value = corrent_count / current_size
            self.log_valid_acrr.record()

            return corrent_count, current_size

        if not end_epoch:
            right, size = handle_datas(datas)
            self.total += size
            self.corret += right
        else:
            logging.info("End a evaling step.")
            self.log_valid_loss.log_current_avg_loss(self.golbal_step)
            self.log_valid_acrr.log_current_avg_loss(self.golbal_step)
            accu = self.corret * 1.0 / self.total
            self.best_accurace = max((self.best_accurace, accu))
            self.total = self.corret = 0

    def __batch_domain_label__(self, batch_size):
        # Generate all Source and Domain label.
        SOURCE = 1
        TARGET = 0

        sd = torch.Tensor(batch_size, 1).fill_(SOURCE)
        td = torch.Tensor(batch_size, 1).fill_(TARGET)

        sd, td, = anpai((sd, td), self.params.use_gpu, False)
        return sd, td
