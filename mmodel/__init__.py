
def get_module(name):
    name = name.upper()
    if name == 'MSDA':
        from .MSDA import params, MSDA
        return params.get_params(), MSDA.Network()
    elif name == 'TADA':
        from .TADA import params, TADA
        return params.get_params(), TADA.TADA()
    elif name == 'DANN':
        from .DANN import params, model
        return params.get_params(), model.DANN()
    elif name == 'MNIST':
        from .MNIST import mnist
        return None, mnist.MNIST()
    elif name == 'FINETUNE':
        from .Finetune import model
        return None, model.Finetune()
    elif name == 'BY':
        from .Bayes import model
        return None, model.BayesModel()
    elif name == 'OPEN':
        from .OpenSet import model
        return None, model.OpensetDA()

def get_params():
    from .basic_params import get_param_parser
    return get_param_parser().parse_args()