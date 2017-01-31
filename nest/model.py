""" Model parent class that describes interaction paradigm for all models"""
import tensorflow as tf
import os
import json
import pickle
import hashlib

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
    def __init__(self, sess, hyperparameters={}, save_dir='./run/'):
        self.sess = sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        self.model_name = "model"
        print("Testing develop")
        # copying so that there are no issues with mutable hyperparameters
        self.hyperparameters = hyperparameters.copy()
        self._get_id()
        # TODO update num_epochs on iterations
        self.num_epochs = 0
        self.save_dir = os.path.abspath(save_dir)
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints/")
        self.summary_dir = os.path.join(self.save_dir, "summary/")
        self.setup_savedirs()
        # self.construct() # removing this call because we'll probably want to scope it instead.
    def _get_id(self):
        """ 'Private Method' that generates a new unique id for the particular model instance"""
        # TODO Test This
        m = hashlib.md5()
        m.update(self.model_name.encode('utf-8' ))
        m.update(pickle.dumps(self.hyperparameters))
        self.model_id = m.hexdigest()
    def setup_savedirs(self):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def setup_summaries(self):
        # deletes the summary directory if it exists
        if tf.gfile.Exists(self.summary_dir):
            tf.gfile.DeleteRecursively(self.summary_dir)
        tf.gfile.MakeDirs(self.summary_dir)
        if self.sess:
            self.merged = tf.summary.merge_all()
            self.writer = tf.train.SummaryWriter(self.summary_dir,self.sess.graph)

    def construct(self):
        """Makes the model structure"""
        pass
    def train_batch(self, X, y):
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
    def save(self):
        """ Saves the current model """
        checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if self.sess:
            print("Saving checkpoint to ", checkpoint_file)
            self.saver.save(sess, str(checkpoint_file),global_step = self.global_step)
    def load(self):
        """ Loads the model """
        # TODO add support for spec
        checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        print("Looking for checkpoint file in: ", checkpoint_dir)
        print("Loading checkpoint from ", checkpoint_file)
        if self.sess:
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

