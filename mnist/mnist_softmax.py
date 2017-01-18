"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import tensorflow as tf
import sys
import os
sys.path.append(os.path.pardir)

from model import Model

flags = tf.app.flags


class MNIST_Softmax(Model):
    def __init__(self, sess):
        super().__init__(sess)
        # setup the softmax graph
        with tf.name_scope('input'):
            self.input = tf.placeholder(tf.float32, [None, 784])
            self.labels = tf.placeholder(tf.float32, [None, 10])
        with tf.name_scope('layer1'):
            self.W = self.create_variable(tf.zeros([784, 10]), 'weights')
            self.b = self.create_variable(tf.zeros([10]), 'bias')
            with tf.name_scope('Wx_plus_b'):
                self.logits = tf.matmul(self.input, self.W) + self.b
                tf.summary.histogram('logits', self.logits)
        with tf.name_scope('cross_entropy'):
            # setup the cross_entropy
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(diff)
                tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.name_scope('train'):
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy, global_step=self.global_step)

        # Setup tester
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        # writer for the model.
        self.saver = tf.train.Saver()

        # setup summary writing
        self.setup_summaries()
    def feed(self, batch_features, batch_labels, iteration):
        super().feed(batch_features, batch_labels)
        summary, acc = self.sess.run([self.merged, self.train_step], feed_dict={self.input: batch_features, self.labels: batch_labels})
        self.writer.add_summary(summary, iteration)#, self.global_step)
        return acc
    def test(self, batch_features, batch_labels):
        super().feed(batch_features, batch_labels)
        return self.sess.run(self.accuracy, feed_dict={self.input: batch_features, self.labels: batch_labels})
    def save(self, model_name='mnist', save_dir='./checkpoints/'):
        super().save(model_name=model_name, save_dir=save_dir)
    def load(self, model_name='mnist', load_dir='./checkpoints/'):
        super().load(model_name=model_name, load_dir=load_dir)
