def get_module(name):
    if name == 'MSDA':
        from .MSDA import params, MSDA
        return params.get_params(), MSDA.Network()
    elif name == 'TADA':
        from .TADA import params, TADA
        return params.get_params(), TADA.TADA()