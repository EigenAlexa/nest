

from abc import ABCMeta, abstractmethod
class HyperQueue(metaclass=ABCMeta):
    """
    A class for queueing up hyperparameters. Implementations
    can range from random search, grid search, or
    Bayesian Search.
    """
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def pop(self):
        """ Returns the next hyperparameter set in the HyperQueue"""
        pass
    @abstractmethod
    def push(self, hyperparams):
        """ Adds an item to the queue"""
        # TODO Determine whether we want to update priority or anything like that
        pass

    @abstractmethod
    def update_priority_fn(self, priority_fn):
        """Updates the priority function that determines the priority used in the queue"""
        # This is particularly useful for Bayesian search that will use the information gained from evaluation to determine what the next step shoudl be
        # TODO determine whether this is the right way to do it or not.
        pass


