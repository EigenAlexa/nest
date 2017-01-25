""" Model parent class that describes interaction paradigm for all models"""
import tensorflow as tf
import os
import json

from enum import IntEnum
from abc import ABCMeta

spec_file_ext = '.spec'

class DataSetType(IntEnum):
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
class Model(metaclass=ABCMeta):
    """ The abstract model class"""
    # TODO move the checkpoint_dir
    def __init__(self, sess, hyperparameters={}, checkpoint_dir='./checkpoints/'):
        self.sess = sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        self.model_name = "model"
        self._get_id()
        self.hyperparameters = hyperparameters
        # TODO update num_epochs on iterations
        self.num_epochs = 0
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        # self.construct() # removing this call because we'll probably want to scope it instead.
    def _get_id(self):
        """ 'Private Method' that generates a new unique id for the particular model instance"""
        # TODO implement
        self.model_id = 0
    def setup_summaries(self, summary_dir='./summary'):
        # deletes the summary directory if it exists
        self.summary_dir = os.path.abspath(summary_dir)
        if tf.gfile.Exists(self.summary_dir):
            tf.gfile.DeleteRecursively(self.summary_dir)
        tf.gfile.MakeDirs(self.summary_dir)

        self.merged = tf.summary.merge_all()
        self.writer = tf.train.SummaryWriter(self.summary_dir,self.sess.graph)
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
    def save(self, model_name=None, save_dir=None):
        """ Saves the current model """

        # Uses object properties in case they don't exist already
        if not model_name:
            model_name = self.model_name
        if not save_dir:
            save_dir = self.checkpoint_dir


        # make directory if it does not yet exist
        if not tf.gfile.Exists(save_dir):
            tf.gfile.MakeDirs(save_dir)
        checkpoint_file = os.path.join(save_dir, model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_file)
        print("Saving checkpoint to ", checkpoint_file)
        self.saver.save(self.sess, str(checkpoint_file) + '.ckpt',global_step = self.global_step)
    def load(self, load_dir=None):
        """ Loads the model """
        # TODO add support for spec
        if not load_dir:
            load_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(load_dir)
        print("Loading checkpoint from ", load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No saved file, starting with freshly initialized parameters')
    def close(self):
        """ Run at the end of the file."""
        self.writer.close()
    def load_from_spec(self, spec_file=None):
        if not spec_file:
            spec_filename = self.model_name + str(self.model_id) + spec_file_ext
            spec_file = os.path.join(self.checkpoint_dir, spec_filename)

        # check if spec_file exists
        if os.path.isfile(spec_file):
            with open(spec_file, 'r') as infile:
                data = json.load(infile)

                # TODO add type checking or confirm that this works
                self.hyperparameters = data['hyperparameters']

                # Load checkpoint and summary directory locations
                self.summary_dir = data['summary_dir']
                self.checkpoint_dir = data['checkpoint_dir']

                # Load num_epochs
                self.num_epochs = data['num_epochs']

                # Load comments
                self.comments = data['comments']
        else:
            print('Spec file loading failed; Using normal loading as initialization')

        # Load the file using the checkpoint_dir as specified above.
        self.load()

    def write_spec(self, comments=None, spec_filename=None):
        """ Write the spec of the model to a file for easy loading """

        # set a filename based on model
        if not spec_filename:
            spec_filename = self.model_name + str(self.model_id) + spec_file_ext
        # Add hyperparameters
        data = {}
        data['hyperparameters'] = self.hyperparameters

        # Add checkpoint and summary directory locations
        data['checkpoint_dir'] = str(self.checkpoint_dir)
        data['summary_dir'] = str(self.summary_dir)

        # Add num_epochs/num_iterations
        data['num_epochs'] = str(self.num_epochs)

        # Add comments on what you tried
        data['comments'] = comments
        print('writing spec')
        spec_file = os.path.join(self.checkpoint_dir, spec_filename)
        with open(spec_file, 'w') as outfile:
            json.dump(data, outfile)

