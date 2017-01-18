import tensorflow as tf
from data import Data
from tensorflow.examples.tutorials.mnist import input_data

class MNIST(Data):
    """ Loads the mnist dataset in the abstraction format"""
    def __init__(self, source='./MNIST_data'):
        super().__init__(source, is_local=True)
        self.all_dataset = input_data.read_data_sets(source, one_hot=True)
        self.dataset = self.all_dataset.train

    def get_batch(self, batch_size, unsupervised=False):
        """Returns a batch of images"""
        if not unsupervised:
            #  Returns a tuple (batch_images, batch_labels)
            return self.dataset.next_batch(batch_size)
        else:
            return self.dataset.next_batch(batch_size)[0]

    def set_dataset_type(self, dataset_type):
        """ sets the type of dataset used by get_batch"""
        if dataset_type == "train":
            self.dataset = self.all_dataset.train
        elif dataset_type == "test":
            self.dataset  = self.all_dataset.test
        elif dataset_type == "validation":
            self.dataset = self.all_dataset.validation
        else:
            raise ValueError("Type only accepts `train, test, or validation`. You input {}".format(dataset_type))
    """ No open/close because these operations there is no need for this dataset"""


