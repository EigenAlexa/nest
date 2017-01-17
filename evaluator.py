""" The evaluation framework that all evaluation frameworks will inherit from """
import tensorflow as tf

class Evalutor:
    def __init__(self, model, data_source):
        """Setup the evaluator"""
        self.model = model
        self.data_source = data_source
    def evaluate(self):
        pass
