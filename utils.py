def hyperdefault(name, default, hyper_dict):
    """
    Handles the default assignment of parameters that may or may
    not appear in the hyperparameter dict. Adds them to the hyperparameter dict
    """
    if name not in hyper_dict:
        hyper_dict[name] = default
    return hyper_dict[name]