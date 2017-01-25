""" The evaluation framework that all evaluation frameworks will inherit from """
import tensorflow as tf
from abc import ABCMeta, abstractmethod
class Evaluator(metaclass=ABCMeta):

    def __init__(self, source, model_store):
        """Setup the evaluator"""
        self.data_source = source

    @abstractmethod
    def evaluate(self, model):
        """
        Passes the data_source through the model and
        compares output to expected output
        :param model: The model to be evaluated
        :return:
        """
        pass
