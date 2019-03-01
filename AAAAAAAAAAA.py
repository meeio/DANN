import itertools
from mmodel.basic_module import DAModule, TrainableModule, ELoaderIter
from mmodel.AAAA.networks import *

from mdata.partial_folder import MultiFolderDataHandler

import torch
from params import get_param_parser
from mmodel.mloger import GLOBAL
import logging

import numpy as np
import itertools


def get_c_param():
    parser = get_param_parser()
    parser.add_argument(
        "--gamma", type=float, default=0.01, help="scaler of entropy loss"
    )
    return parser.parse_args()


def entropy(inputs):
    """given a propobility inputs in range [0-1], calculate entroy
    
    Arguments:
        inputs {tensor} -- inputs
    
    Returns:
        tensor -- entropy
    """

    def entropy(p):
        return -1 * p * torch.log(p)

    e = entropy(inputs) + entropy(1 - inputs)

    return e


STAGE = {"adaptation": 0, "prediction": 1, 'training' : 3}


class PredictUnit(TrainableModule):
    def __init__(
        self,
        turn_key,
        hold_key,
        classes_map,
        params,
    ):

        super(PredictUnit, self).__init__(params)

        self._turn_key = turn_key
        self.hold_key = hold_key

        # resnet50 and it's avepool
        self.params = params

        # self.domain = domain
        # self.idx = predict_classes_idx
        self.classes_to_idx = classes_map

        self.output_shape = None
        self.datas = None

        self.steps = 1

        self.predict = None
        self.global_atten = None

        self.creterion = torch.nn.BCELoss()

        self._all_ready()

    @property
    def turn_key(self):
        return self._turn_key[0]

    def set_data(self, inputs):
        stage, datas = inputs
        self.current_stage = stage
        self.current_data = datas

    def _prepare_data(self):
        return None, None

    def _feed_data(self, mode):

        stage = self.current_stage
        
        if stage == STAGE['adaptation']:
            if mode == 'train':
                feature, domain = self.current_data
                return feature, domain

        elif stage == STAGE['prediction']:
            pass
        
        elif stage == STAGE['training']:
            pass
        
        else:
            raise Exception('Stage Error')

        batch_size = self.datas[0].size()[0]
        domain = 1 if self.datas[1] is "S" else 0
        label = torch.Tensor(batch_size, 1).fill_(domain)
        self.datas[1] = label
        return self.datas

    def _regist_networks(self):

        regist_dict = {
            "l_D_" + str(i): SmallDomainClassifer() for i in range(49)
        }
        regist_dict["C"] = Classifier(len(self.classes_to_idx))
        regist_dict["g_D"] = DomainClassifier()

        regist_dict["avgpool"] = nn.AvgPool2d(7, stride=1)

        return regist_dict

    def _regist_losses(self):

        local_dis_nets = ["l_D_" + str(i) for i in range(49)]
        self.regist_loss("local_dis", *local_dis_nets)

        self.regist_loss("global_dis", "g_D")

        self.regist_loss("classifer", "C")

    def _local_dis(self, features):
        """ given a feature mask, producing it's local attention        mask.
        
        Arguments:
            features {tensor} -- given feature
        
        Returns:
            tensor -- loal feature mask
        """

        # [batch_size, c, h , w]
        size = features.size()
        # [batch_szie, h * w]
        batch = size[0]
        spatial_size = size[2] * size[3]
        features = features.view(batch, -1, spatial_size)

        result = torch.zeros([batch, spatial_size])
        for i in range(spatial_size):
            inputs = features[:, :, i]
            local_dis = self.networks["l_D_" + str(i)](inputs)
            local_dis = local_dis.squeeze()
            result[:, i] = local_dis

        return result

    def _stage(self, stage, datas, is_training):

        assert stage in STAGE

        #########################################
        ## begain stage one
        ## in the stage one local_atten and global atten are calculated
        #########################################
        if stage == STAGE["adaptation"]:

            features, domain_label = datas

            # calculate local and glboal attention
            # save as class propertiy

            # calculate local attention based on l_atten = 2 - H(D(f))
            l_domain_predict = self._local_dis(features)
            l_atten = entropy(l_domain_predict)
            l_atten = 2 - l_atten  # 1 + (1 - l_atten)

            # resize to size of feature
            size = features.size()
            l_atten = l_atten.view(size[0], size[2], size[3])
            l_atten = l_atten.unsqueeze(1)
            l_atten = l_atten.expand(size)

            # reweight feature based on local attention
            features = features * l_atten

            # go throught bottleneck layers
            g_features = self.avgpool(features)

            # cal global attention and make prediction
            g_domain_predict = self.g_D(g_features)
            g_atten = entropy(g_domain_predict)
            g_atten = 1 + g_atten

            self.attentions = (l_atten, g_atten)

            # if need trainning
            # update local domain discriminator
            # and global domain discriminator
            if self.turn_key == self.hold_key and is_training:

                # resize local predict result to [b*49, 1]
                l_domain_predict = l_domain_predict.view(-1, 1)

                # domain label is tensor of size [1,1]
                # repeat it to size of [b*49, 1]
                l_domain_label = domain_label.repeat(
                    l_domain_predict.size()
                )

                l_D, l_G = self._adversiral_loss(
                    l_domain_predict, domain_label
                )

                # perform the same process for global dis as local dis
                g_domain_label = domain_label.repeat(
                    g_domain_predict.size()
                )

                g_D, g_G = self._adversiral_loss(
                    g_domain_predict, g_domain_label
                )

                return l_D, g_D, l_G, g_G
            return None

        #########################################
        ## begain stage two
        ## in the stage tow, classes prediction are made
        #########################################
        elif stage == STAGE['prediction']:
            features, local_atten = datas
            # retrieval previous features
            features = features * local_atten
            g_features = self.avgpool(features)
            predict_result = self.C(g_features)
            self.predict = predict_result

        #########################################
        ## begain stage two
        ## in the stage tow, classes prediction are made
        #########################################
        elif stage == STAGE['updating']:
            assert is_training
            l_D, g_D = self.dis_loss
            self._update_loss("local_dis", l_D)
            self._update_loss("global_dis", g_D)

    def _train_process(self, datas, **kwargs):
        stage = self.current_stage

        if stage == STAGE["adaptation"]:
            if self.turn_key == self.hold_key:
                l = self._stage(stage, datas, is_training=True)
                self.losses = l
            else:
                self._stage(stage, source_data, is_training=True)

        elif stage == STAGE['prediction']:
            pass
        elif stage == STAGE['updating']:
            pass

    def _finish_a_train_process(self):
        self.golbal_step += 0.5

    def _log_process():
        return

    def _eval_process(self, datas, **kwargs):
        stage = kwargs["stage"]
        self._stage(stage, datas, is_training=False)

    def _adversiral_loss(self, prediton, original_domain):
        creterion = self.creterion

        loss_D = creterion(prediton, original_domain)

        # original gan loss will be:
        # loss_G = bce_loss(source_predict, TARGET_DOMAIN)
        # loss_G = bce_loss(target_predict, SOURCE_DOMAIN)
        other_domain = torch.abs(original_domain - 1)
        loss_G = creterion(prediton, original_domain)
        +creterion(prediton, other_domain)
        loss_G = 0.5 * loss_G

        return loss_D, loss_G

    def result(self):
        return self.global_atten, self.predict

    def __str__(self):
        return "Unit of %s in group %d " % (self.domain, self.idx)


class Network(TrainableModule):
    def __init__(self, params):

        super(Network, self).__init__(params)

        # init multi source image folder handler
        mhandler = MultiFolderDataHandler(
            root="Office_Shift",
            sources=["A", "W"],
            target="D",
            params=params,
        )

        # get all class separations
        # ics is a list of ClassSeperation
        # every ClassSeperation is consist of (classes) and (domains)
        ics = mhandler.independ_class_seperation

        unit_order = [
            (idx, domain)
            for idx, sep in enumerate(ics)
            for domain in sep.domains
        ]

        self.mhandler = mhandler
        self.independ_class_seperation = ics
        self.unit_order = unit_order

        # set group iterator
        class_group_idx = [i for i in range(len(ics))]
        self.group_idx_iter = itertools.cycle(iter(class_group_idx))
        self.class_group_idxs = class_group_idx


        self.CE = nn.CrossEntropyLoss()

        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr
        )

        self._turn_key = [None]
        self._all_ready()

    @property
    def turn_key(self):
        return self._turn_key[0]

    @turn_key.setter
    def turn_key(self, turn_key):
        assert len(turn_key) == 2
        self._turn_key[0] = turn_key

    def _iter_all_unit(self, func, **kwargs):
        for idx, domain in self.unit_order:
            func(idx, domain, **kwargs)

    def _regist_networks(self):

        F = FeatureExtroctor(self.params)

        classes_sep = [i.classes_to_idx for i in self.independ_class_seperation]        
        unions = [{} for i in range(len(self.independ_class_seperation))]
        def create_predict_unit(idx, domain):
            hold_key = (idx, domain)
            unions[idx][domain] = PredictUnit(
                self._turn_key, hold_key, classes_sep[idx], self.params            
            )
        self._iter_all_unit(create_predict_unit)
        self.unions = unions

        return {"F": F}

    def _regist_losses(self):

        n = [self.networks["F"]]
        self.regist_loss("prediction", *n)

    def _prepare_data(self):
        """ return dataloader.
        
        Returns:
            list -- datainfo, train_loader, valid_loader

            datainfo is dict conatins data info pairs

            train_loader is a tuple:
                first item is source data loaders whic is a dict of
                form {group_idx : {domain : dataloader}}

                second item is dataloader of target domain.
            
            valid loader is same as target train loader.
        """

        data_info = None

        icgs, target_loader, valid_loader = (
            self.mhandler.seperation_with_loader()
        )

        # return all iters
        iters = dict()

        # get dataloader in orders
        partial_source_loaders = [
            {} for i in range(len(self.independ_class_seperation))
        ]
        def unit_loader(idx, domain):
            l = icgs[idx].get_domain_loader(domain)
            it = ELoaderIter(l)
            partial_source_loaders[idx][domain] = it
        self._iter_all_unit(unit_loader)

        # set trainging iters
        mode = "train"
        iters[mode] = dict()
        iters[mode]["target"] = ELoaderIter(target_loader)
        iters[mode]["source"] = partial_source_loaders

        # set validing iters
        mode = "valid"
        loader = valid_loader
        iters[mode] = ELoaderIter(loader)

        return data_info, iters

    def _feed_data(self, mode):

        if mode == "train":
            iters = self.iters[mode]["source"]
            target_iters = self.iters[mode]["target"]
                        
            feed_datas = list()
            def feeded_datas(idx, domain):
                img, label = iters[idx][domain].next()
                feed_datas.append(img)
                feed_datas.append(label)
            self._iter_all_unit(feeded_datas)

            feed_datas.append(target_iters.next()[0])
            return feed_datas
        else:
            return self.iters[mode].next(need_end=True)

    def _train_process(self, datas, **kwargs):

        orders = self.unit_order
        union_range = range(len(self.independ_class_seperation))

        source_domain_num = len(orders)
        source_datas = [
            (datas[i], datas[i + 1])
            for i in range(0, source_domain_num * 2, 2)
        ]

        t_img = datas[-1]
        target_data = ((t_img, None),)

        def get_losses_from(datas, domain):
            
            confuse_loss = list()
            predict_loss = list()

            domain_label = torch.Tensor(1,1).fill_(domain)

            for idx, (s_img, s_label) in enumerate(datas):
                
                # get source features
                feature = self.F(s_img)

                #########################################
                ## Update turn_key, if len of unions is 3
                ## then trun_key will change from 0-2
                ## 
                ## Notice that all unit will get the key only ONCE
                ## in this batch of data
                #########################################
                sep_id, domain = orders[idx]
                self.turn_key = (sep_id, domain)

                #########################################
                ## perfoorming adaptation stage
                ##
                ## In this stage, all attention will be calculated.
                ##
                ## Loss function will be saved in unit when the key is 
                ## right.
                #########################################
                inputs = (STAGE["adaptation"], (feature, domain_label))
                l_attens = [{} for i in union_range]     
                g_attens = [{} for i in union_range]

                def stage_adaptation(idx, domain):
                    unit = self.unions[idx][domain]
                    if s_label is None:
                        self.turn_key = unit.hold_key
                    unit.set_data(inputs)
                    unit.train_module()
                    l, g = unit.attentions
                    l_attens[idx][domain] = l
                    g_attens[idx][domain] = g

                self._iter_all_unit(stage_adaptation)

                #########################################
                ## perfoorming classify stage
                ##
                ## In this stage, new local attention and original feature
                ## map are feeded into unit
                ##
                ## the classify result will be calculated
                #########################################

                predict = [{} for i in union_range]

                
                def stage_classify(idx, domain):
                    unit = self.unions[idx][domain]
                    unit.set_data(inputs)
                    unit.train()
                    predict[idx][domain] = unit.result()

            return confuse_loss, predict_loss  

        s_confuse, s_predict = get_losses_from(source_datas, domain=1)
        t_confuse, t_predict = get_losses_from(target_data, domain=0)

    def _perform_stage_one(self, datas, train=False):

        func_name = "train_module" if train else "eval_module"

        def run_sub_module(e, **kargs):
            getattr(e, func_name)(**kargs)

        # stage one

        for union in self.unions:
            for domain in union:
                unit = union[domain]
                unit.set_data(features)
                run_sub_module(unit, stage="one")

    def _make_prediction(self, features, train=True):
        """ After a process, make predition from all union
        
        Arguments:
            datas {tuple} -- (img, (group_idx, domain))
        
        Keyword Arguments:
            train {bool} -- is trianning process (default: {True})
        
        Returns:
            tensor -- prediction result
        """

        func_name = "train_module" if train else "eval_module"

        def run_sub_module(e, **kargs):
            getattr(e, func_name)(**kargs)

        # stage one
        for union in self.unions:
            local_attens = list()
            for domain in union:
                unit = union[domain]
                unit.set_data(features)
                run_sub_module(unit, stage="one")

        # stage two
        union_result = list()
        for union in self.unions:
            attens = list()
            predictions = list()
            for domain in union:
                unit = union[domain]
                # unit.set_data(datas)
                run_sub_module(unit, stage="two")
                g_domain_dis, prediction = unit.result()
                predictions.append(prediction)

            weighted_predict = sum(
                attens[i] * predictions[i] for i in range(len(predictions))
            ) / sum(attens)
            union_result.append(weighted_predict)

        return torch.cat(union_result, 1)

    def _eval_process(self, datas, **kwargs):
        return

    def _log_process(self):
        return


if __name__ == "__main__":

    params = get_c_param()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
    )

    n = Network(params)
    n.train_module()

    # nadd = MANN(params)
    # nadd.train()

    # from torchvision import models
    # from torchsummary import summary

    # feature = FeatureExtractor()
    # feature.weight_init()
    # feature.cuda()
    # summary(feature, (3, 32, 32))
