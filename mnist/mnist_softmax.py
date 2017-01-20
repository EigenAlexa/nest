"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import tensorflow as tf

from model import Model

flags = tf.app.flags


class MNISTSoftmax(Model):
    def __init__(self, sess):
        super().__init__(sess)
        # setup the softmax graph

        with tf.name_scope('mnist_softmax'):
            self.construct()
        # writer for the model.
        self.saver = tf.train.Saver()
        self.model_name = 'mnist'
        # setup summary writing
        self.setup_summaries()
    def construct(self):
        super().construct()
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

    def feed(self, batch_features, batch_labels, iteration, execution_stats=False):
        super().feed(batch_features, batch_labels)

        if execution_stats:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = self.sess.run([self.merged, self.train_step], feed_dict={self.input: batch_features, self.labels: batch_labels}, options=run_options, run_metadata=run_metadata)
            self.writer.add_run_metadata(run_metadata, 'step{}'.format(iteration))
        else:
            summary, _ = self.sess.run([self.merged, self.train_step], feed_dict={self.input: batch_features, self.labels: batch_labels})

        self.writer.add_summary(summary, iteration)#, self.global_step)
    def test(self, batch_features, batch_labels):
        super().feed(batch_features, batch_labels)
        summary, acc = self.sess.run([self.merged, self.accuracy], feed_dict={self.input: batch_features, self.labels: batch_labels})
        return acc
