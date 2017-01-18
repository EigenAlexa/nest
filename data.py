"""The Data Parent Class that all dataset loaders will inherit from."""
import tensorflow as tf

class Data:
    def __init__(self, source, is_local=False):
        """ Setups a python interface feedign from the database source """
        pass
    def get_batch(self, batch_size):
        """ Returns a batch of data. Classes that inherit need to specify whether this data includes labels or not. """
        pass
    def open(self):
        """ Opens the data source """
        pass
    def close(self):
        """ Closes the data source """
        pass
