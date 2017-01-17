"""The Data Parent Class that all dataset loaders will inherit from."""
import tensorflow as tf

class Data:
    def __init__(self, source, is_local=False):
        """ Setups a python interface feedign from the database source """
        pass
    def next_train_batch(self, batch_size):
        """ Returns a placeholder for the next training batch """
        pass
    def next_test_batch(self, batch_size):
        """ Returns a placeholder for the next testing batch """
        pass
    
