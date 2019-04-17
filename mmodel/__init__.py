
def get_module(name):
    name = name.upper()
    if name == 'MSDA':
        from .msda import params, MSDA
        return params.get_params(), MSDA.Network()
    elif name == 'TADA':
        from .tada import params, TADA
        return params.get_params(), TADA.TADA()
    elif name == 'DANN':
        from .dann import params, model
        return params.get_params(), model.DANN()
    elif name == 'FINETUNE':
        from .fine_tune import model
        return None, model.Finetune()
    elif name == 'BY':
        from .bayes import model
        return model.param, model.BayesModel()
    elif name == 'OPEN':
        from .openset import model
        return model.param, model.OpensetDA()
    elif name == 'OPENBB':
        from .openset_by_backprop import oldmodel
        return oldmodel.param, oldmodel.OpensetBackprop()
    elif name == 'OPENBBTEST':
        from .openset_by_backprop import model
        return model.param, model.OpensetBackprop()
    elif name == 'OPENDP':
        from .openset_drop import model
        return model.param, model.OpensetDrop()
    elif name == 'PDP':
        from .openset_drop import model_partial
        return model_partial.param, model_partial.PartialDrop()


def get_params():
    from .basic_params import get_param_parser
    return get_param_parser().parse_args()