"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import tensorflow as tf
import sys
import os
sys.path.append(os.path.pardir)

from model import Model

FLAGS = None

class MNIST_Softmax(Model):
    def __init__(self, sess):
        super().__init__(sess)
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b= tf.Variable(tf.zeros([10]))
        self.logits = tf.matmul(self.input, self.W) + self.b
        self.labels = tf.placeholder(tf.float32, [None, 10])

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy, global_step=self.global_step)

        # Setup tester
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # writer for the model.
        self.saver = tf.train.Saver()
    def feed(self, batch_features, batch_labels):
        super().feed(batch_features, batch_labels)
        self.sess.run(self.train_step, feed_dict={self.input: batch_features, self.labels: batch_labels})
    def test(self, batch_features, batch_labels):
        super().feed(batch_features, batch_labels)
        return self.sess.run(self.accuracy, feed_dict={self.input: batch_features, self.labels: batch_labels})
    def save(self, dir='./',model_name='mnist'):
        super().save(dir, model_name)
    def load(self, dir='./', model_name='mnist'):
        super().load(dir, model_name)








