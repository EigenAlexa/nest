from trainer import Trainer
from mnist_hyperqueue import MNISTHyperQueue
class MNISTTrainer(Trainer):
    """
    Trains MNISTSotmax regression. Does hyperparameter search.
    """
    # TODO figure out how to implement stopping points
    # probably some measurement of cost, accuracy, epochs, grid_searches
    def __init__(self, model_class, evaluator, hyper_queue=None, max_instances=1):
        if not hyper_queue:
            hyper_queue = MNISTHyperQueue()
        super().__init__(model_class, evaluator, hyper_queue, max_instances)
