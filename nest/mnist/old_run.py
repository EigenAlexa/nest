# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import argparse
import sys
from mnist_softmax import MNISTSoftmax
from mnist_data import MNISTData
from model import DataSetType

import tensorflow as tf

FLAGS = None
def main(_):
    # Import data
    sess = tf.InteractiveSession()
    print("Loading MNIST")
    data = MNIST()
    print("MNIST Loaded")
    print("Setting up Model")
    model = MNISTSoftmax(sess)
    model.load()


    print("Variable setup")
    tf.global_variables_initializer().run()
    # Train
    for i in range(1000):
        batch_xs, batch_ys = data.get_batch(100)

        if i % 50 == 0:
            data.set_dataset_type(DataSetType.TEST)
            acc = model.test(*data.get_batch(1000))
            print('Iteration {}; Accuracy {}'.format(i, acc))
            data.set_dataset_type(DataSetType.TRAIN)
        else:
            acc = model.feed(batch_xs, batch_ys, i, execution_stats=(i%100 == 99))
        # Test trained model
    print('Testing')

    model.save()
    model.write_spec("this is a dope comment")
    model.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
