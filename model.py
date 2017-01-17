""" Model parent class that describes interaction paradigm for all models"""
import tensorflow as tf

class Model:
    def __init__(self):
        pass
    def train(self, data_source):
        """Trains the model"""
        pass
    def feed(self, batch):
        pass
