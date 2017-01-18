""" Model parent class that describes interaction paradigm for all models"""
import tensorflow as tf
import os

class Model:
    def __init__(self, sess):
        self.sess = sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        pass
    def train(self, data_source):
        """Trains the model"""
        pass
    def feed(self, batch_features, batch_labels):
        """Passes the data throug the model"""
        pass
    def test(self, batch_features, batch_labels):
        pass
    def save(self, dir='./', model_name='model'):
        """ Saves the current model """
        checkpoint_file = os.path.abspath(dir + model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_file)
        print("Saving checkpoint to ", checkpoint_file)
        self.saver.save(self.sess, dir + model_name + '.ckpt',global_step = self.global_step)
    def load(self, dir='./', model_name='model'):
        checkpoint_file = os.path.abspath(dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_file)
        print("Loading checkpoint from ", checkpoint_file)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No saved file, starting with freshly initialized weights')

