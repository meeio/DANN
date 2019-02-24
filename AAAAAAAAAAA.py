import itertools
from mmodel.basic_module import DAModule, TrainableModule, ELoaderIter
from mmodel.AAAA.networks import *

from mdata.partial_folder import MultiFolderDataHandler

import torch
from params import get_param_parser
from mmodel.mloger import GLOBAL
import logging

import numpy as np


def get_c_param():
    parser = get_param_parser()
    parser.add_argument(
        "--gamma", type=float, default=0.01, help="scaler of entropy loss"
    )
    return parser.parse_args()


def domain_confusion_loss(predicted_domain):
    """ calculate domain confusion loss
    
    Arguments:
        predicted_domain {tensor} -- predicted domain
    
    Returns:
        tensor -- tensor of size[[]] 
    """

    loss = torch.log(predicted_domain) + torch.log(1 - predicted_domain)

    loss = torch.mean(loss)
    return loss


def entropy(inputs, reduce_mean=True, make_binary=False):
    """given a propobility inputs in range [0-1], calculate entroy
    
    Arguments:
        inputs {tensor} -- inputs
    
    Returns:
        tensor -- entropy
    """

    def entropy(p):
        return -1 * p * torch.log(p)

    if make_binary:
        e = entropy(inputs) + entropy(1 - inputs)
    else:
        e = entropy(inputs)

    return torch.mean(e) if reduce_mean else e


class PredictUnit(TrainableModule):
    def __init__(
        self,
        feature_extractor,
        predict_classes_idx,
        domain,
        classes_map,
        params,
    ):

        super(PredictUnit, self).__init__(params)

        # resnet50 and it's avepool
        self.F = feature_extractor
        self.params = params

        self.domain = domain
        self.idx = predict_classes_idx
        self.classes_to_idx = classes_map
        self.need_trainning = False

        self.output_shape = None
        self.datas = None

        self.steps = 1

        self.predict = None
        self.global_atten = None

        self._all_ready()

    def set_data(self, datas):
        self.datas = datas[0]

        current_idx, current_domain = datas[1]
        self.need_trainning = (current_idx == self.idx) and (
            current_domain == self.domain
        )

    def _prepare_data(self):
        return None, None

    def _feed_data(self, mode):
        return self.datas

    def _regist_networks(self):

        output_size = self.F.output_size()

        regist_dict = {
            "l_D_" + str(i): SmallDomainClassifer() for i in range(49)
        }
        regist_dict["C"] = Classifier(len(self.classes_to_idx))
        regist_dict["g_D"] = DomainClassifier()
        regist_dict["F"] = self.F

        regist_dict["avgpool"] = nn.AvgPool2d(7, stride=1)

        return regist_dict

    def _regist_losses(self):

        for name, network in self.networks.items():
            if name.startswith("l_D"):
                i = name.split("_")[-1]
                log = self.params.log
                self.params.log = False
                # self.regist_loss("local_dis_" + i, name)
                self.regist_loss("local_confuse_" + i, name, "F")
                self.params.log = log

            elif name == "g_D":
                # self.regist_loss("global_dis", name)
                self.regist_loss("global_confuse", name, "F")

            elif name == "C":
                self.regist_loss("classifer", name)

    def _local_attention_mask(self, features):
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

    def _train_process(self, datas, **kwargs):
        img = datas

        print(kwargs)
        print(stop)

        features = self.F(img)
        l_domain_predict = self._local_attention_mask(features)

        # calculate local attention based on l_atten = 2 - H(D(f))
        l_atten = entropy(l_domain_predict, reduce_mean=False)
        l_atten = 2 - l_atten

        # resize to size of feature
        size = features.size()
        l_atten = l_atten.view(size[0], size[2], size[3])
        l_atten = l_atten.unsqueeze(1)
        l_atten = l_atten.expand(size)

        # reweight feature attention
        features = features * l_atten

        # go throught bottleneck layers
        g_features = self.avgpool(features)

        # cal global attention and make prediction
        g_domain_predict = self.g_D(g_features)
        g_atten = entropy(g_domain_predict, reduce_mean=False)
        predict_result = self.C(g_features)

        # print(g_atten)
        # print(stop)
        # if need trainning domain discriminator
        if self.need_trainning:
            # update local domain discriminator
            lenth = l_domain_predict.size()[-1]
            for i in range(lenth):
                l = domain_confusion_loss(l_domain_predict[:, -1])
                self._update_loss("local_confuse_" + str(i), l)

            # update global domain discriminator
            l = domain_confusion_loss(g_domain_predict)
            self._update_loss("global_confuse", l)

        self.predict = predict_result
        self.global_atten = g_domain_predict

    def _finish_a_train_process(self):
        self.golbal_step += 0.5

    def _log_process():
        return

    def _eval_process(self, datas, **kwargs):
        return

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

        ics = mhandler.independ_class_seperation
        # get all class separations
        self.mhandler = mhandler
        # ics is a list of ClassSeperation
        # every ClassSeperation is consist of (classes) and (domains)
        self.independ_class_seperation = ics

        # set group iterator
        class_group_idx = [i for i in range(len(ics))]
        self.group_idx_iter = itertools.cycle(iter(class_group_idx))
        self.class_group_idxs = class_group_idx
        self.current_group_idx = 0
        self.current_domain = None
        self.current_domain_iter = None

        self.unions = list()
        self.all_classifers = list()

        self.CE = nn.CrossEntropyLoss()

        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr
        )

        self._all_ready()

    def _regist_networks(self):

        F = FeatureExtroctor(self.params)

        ics = self.independ_class_seperation
        params = self.params
        unions = list()
        # for an union
        for i, seperation in enumerate(ics):
            cti = ics[i].classes_to_idx
            union = dict()
            # for an unit
            for domain in seperation.domains:
                unit = PredictUnit(F, i, domain, cti, params)
                union[domain] = unit
                self.all_classifers.append(unit.C)
            unions.append(union)
        self.unions = unions

        return {"F": F}

    def _regist_losses(self):

        n = [self.networks["F"]] + self.all_classifers
        self.regist_loss("prediction", *n)

    def _make_unit():
        output_shape = None
        C = Classifer(params, c * h * w)

        local_Ds = list()
        for i in range(output_shape):
            D = SmallDomainClassifer()
            local_Ds.append(D)

        global_D = DomainClassifier()

        return global_D, local_Ds, C

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

        # get dataloader inorders
        partial_loaders = list()
        for seperation in icgs:
            domain_iter = dict()
            for domain in seperation.domains:
                domain_iter[domain] = seperation.get_domain_loader(domain)
            partial_loaders.append(domain_iter)

        # return all iters
        iters = dict()

        mode = "train"
        iters[mode] = dict()
        iters[mode]["target"] = ELoaderIter(target_loader)
        iters[mode]["source"] = list()
        for idx, domain_loaders in enumerate(partial_loaders):
            domain_iters = {
                d: ELoaderIter(l) for d, l in domain_loaders.items()
            }
            iters[mode]["source"].append(domain_iters)

        mode = "valid"
        loader = valid_loader
        iters[mode] = ELoaderIter(loader)

        return data_info, iters

    def _feed_data(self, mode):

        if mode == "train":
            # end_group = self.current_domain_group[-1] == self.current_domain
            try:
                self.current_domain = next(self.current_domain_iter)
            except Exception:
                idx = next(self.group_idx_iter)
                self.current_group_idx = idx
                domain_group = list(self.iters[mode]["source"][idx].keys())
                self.current_domain_iter = iter(domain_group)
                self.current_domain = next(self.current_domain_iter)

            s_iters = self.iters[mode]["source"]
            cs_iters = s_iters[self.current_group_idx][self.current_domain]

            s_img, s_label = cs_iters.next()
            t_img, _ = self.iters[mode]["target"].next()

            return s_img, s_label, t_img

        else:
            return self.iters[mode].next(need_end=True)

    def _train_process(self, datas, **kwargs):

        s_img, s_label, t_img = datas

        group_idx = self.current_group_idx
        domain = self.current_domain

        def _process(img, label=None):
            # handle source data
            datas = (img, (group_idx, domain))
            prediction = self._make_prediction(datas, train=True)
            preidction_loss = self.params.gamma * entropy(
                prediction, make_binary=False
            )
            if label is not None:
                preidction_loss += self.CE(prediction, s_label)

            print(preidction_loss)
            self._update_loss("prediction", preidction_loss)

        prediction = _process(s_img, s_label)
        prediction = _process(t_img)

    def _make_prediction(self, datas, train=True):
        """ After a process, make predition from all union
        
        Arguments:
            datas {tuple} -- (img, (group_idx, domain))
        
        Keyword Arguments:
            train {bool} -- is trianning process (default: {True})
        
        Returns:
            tensor -- prediction result
        """

        func_name = 'train_module' if train else 'eval_module'
        def make_procedure(e, **kargs):
            getattr(e, func_name)(**kargs)



        # stage one
        union_result = list()
        for union in self.unions:
            local_attens = list()
            for domain in union:
                unit = union[domain]
                unit.set_data(datas)
                make_procedure(unit, stage='one')
                


        for union in self.unions:
            attens = list()
            predictions = list()
            for domain in union:
                unit = union[domain]
                unit.set_data(datas)
                make_procedure(unit, stage="two")
                g_domain_dis, prediction = unit.result()
                attens.append(entropy(g_domain_dis, make_binary=True))
                predictions.append(prediction)
            weighted_predict = sum(
                attens[i] * predictions[i] for i in range(len(predictions))
            ) / sum(attens)
            union_result.append(weighted_predict)

        result = torch.cat(union_result, 1)
        return result

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

