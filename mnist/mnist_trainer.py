from trainer import Trainer

class MNISTTrainer(Trainer):
    """
    Trains MNISTSotmax regression. Does hyperparameter search.
    """
    # TODO figure out how to implement stopping points
    # probably some measurement of cost, accuracy, epochs, grid_searches
    def __init__(self, model_class, evaluator, ):
        pass