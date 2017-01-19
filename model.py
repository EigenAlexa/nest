""" Model parent class that describes interaction paradigm for all models"""
import tensorflow as tf
import os

from enum import Enum

class DataType(Enum):
     TRAIN = 0
     TEST = 1
     VALIDATION = 2
     
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # TODO add options to disable certain summaries.
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
class Model:
    def __init__(self, sess):
        self.sess = sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        # self.construct() # removing this call because we'll probably want to scope it instead.
    def setup_summaries(self, summary_dir='./summary'):
        # deletes the summary directory if it exists
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)
        tf.gfile.MakeDirs(summary_dir)

        self.merged = tf.summary.merge_all()
        self.writer = tf.train.SummaryWriter(summary_dir,self.sess.graph)
    def construct(self):
        """Makes the model structure"""
        pass
    def train(self, data_source):
        """Trains the model"""
        pass
    def add_summary(self, variable):
        variable_summaries(variable)
    def create_variable(self, init, name=None):

        var = tf.Variable(init,name=name)
        self.add_summary(var)
        return var
    def feed(self, batch_features, batch_labels):
        """Passes the data throug the model"""
        pass
    def test(self, batch_features, batch_labels):
        pass
    def save(self, model_name='model', save_dir='./save/'):
        """ Saves the current model """
        if not tf.gfile.Exists(save_dir):
            tf.gfile.MakeDirs(save_dir)
        checkpoint_file = os.path.abspath(save_dir + model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_file)
        print("Saving checkpoint to ", checkpoint_file)
        self.saver.save(self.sess, str(checkpoint_file) + '.ckpt',global_step = self.global_step)
    def load(self, model_name='model', load_dir='./save/',):
        checkpoint_file = os.path.abspath(load_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_file)
        print("Loading checkpoint from ", checkpoint_file)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No saved file, starting with freshly initialized parameters')
    def close(self):
        """ Run at the end of the file."""
        self.writer.close()
