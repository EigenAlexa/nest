import tensorflow as tf
from datasource import DataSource
from model import DataSetType
from tensorflow.examples.tutorials.mnist import input_data

class MNIST(DataSource):
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
        if dataset_type == DataSetType.TRAIN:
            self.dataset = self.all_dataset.train
        elif dataset_type == DataSetType.TEST:
            self.dataset  = self.all_dataset.test
        elif dataset_type == DataSetType.VALIDATION:
            self.dataset = self.all_dataset.validation
        else:
            raise ValueError("Type only accepts those values listed in the model.DataSetType enum variable. You input {}".format(dataset_type))
    """ No open/close because these operations there is no need for this dataset"""


