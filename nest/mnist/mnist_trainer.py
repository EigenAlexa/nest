from nest.trainer import Trainer
from nest.mnist.mnist_hyperqueue import MNISTHyperQueue
import tensorflow as tf
class MNISTTrainer(Trainer):
    """
    Trains MNISTSotmax regression. Does hyperparameter search.
    """
    # TODO figure out how to implement stopping points
    # probably some measurement of cost, accuracy, epochs, grid_searches
    def __init__(self, model_class, data_source, evaluator, model_store, hyper_queue=None, max_instances=1):
        if not hyper_queue:
            hyper_queue = MNISTHyperQueue()
        super().__init__(model_class, data_source, evaluator, model_store, hyper_queue, max_instances)
        self.cur_id = 0

    def start_new_instance(self, hyperparameters):
        """ This is an extension of the api proposed in trainer.py
        however I think it's problematic in the fact that it doesn't
        _really_ distribute anything. However, it is a test to see whether the api
        works more or less"""
        id = self.cur_id
        self.cur_id += 1
        print("Training new instance")
        assert id not in self.training_model_ids, "Duplicate id found, aborting"
        self.training_model_ids.add(id)
        with tf.Session() as sess:

            self.cur_model = self.model_class(sess, hyperparameters, name='mnist'+str(id))
            tf.global_variables_initializer().run()
            acc = 0
            for i in range(1000):
                batch_xs, batch_ys = self.data_source.get_batch(100)

                if i % 50 == 0:
                    acc = self.evaluator.evaluate(self.cur_model)
                    print('Iteration {}; Test Accuracy {}'.format(i, acc))
                else:
                    self.cur_model.feed(batch_xs, batch_ys, i, execution_stats=(i % 100 == 99))
            print('Final Accuracy {}'.format(acc))
            self.done_training(id)
    def done_training(self, model_id):
        tf.reset_default_graph()
        super().done_training(model_id)
