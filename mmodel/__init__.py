
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
        return None, model.BayesModel()
    elif name == 'OPEN':
        from .openset import model
        return None, model.OpensetDA()
    elif name == 'OPENBB':
        from .openset_by_backprop import oldmodel
        return None, oldmodel.OpensetBackprop()
    elif name == 'OPENDP':
        from .openset_drop import model
        return None, model.OpensetDrop()


def get_params():
    from .basic_params import get_param_parser
    return get_param_parser().parse_args()