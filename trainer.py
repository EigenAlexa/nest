import tensorflow as tf
from abc import ABCMeta, abstractmethod

class Trainer(metaclass=ABCMeta):

    def __init__(self, model_class, evaluator, model_store, hyper_queue, max_instances):
        self.model_class = model_class
        self.evaluator = evaluator
        self.model_store = model_store
        self.hyper_queue = hyper_queue
        self.training_model_ids = set()
        self.max_instances = max_instances

    def start_new_instance(self, hyperparameters):
        """
        Takes in a set of hyperparameters, initializes  and starts training a new instance of model_class
        :return: model_id for new instance
        """

        # TODO implement
        # get id
        id = 0 # TODO acutally get the id
        self.training_model_ids.add(id)
        raise NotImplementedError()
    def continue_training_instance(self, model_id):
        """
        Loads in a pretrained model from model_id and continues training """
        # TODO implement
        raise NotImplementedError()
        pass
    def evaluate_instance(self, model):
        """ Evaluates a model instance. """
        # Takes in the model, grabs the id and passes that to the evaluator.
        # The evaluator then pulls the model files from the model server,
        # reconstructs the graph, loads the checkpoint, and
        # TODO implement

        raise NotImplementedError()
        pass
    def is_still_searching(self):
        """
        :return: Boolean whether the training method should stop
        """
        # TODO figure out what should be done here
        pass
    def done_training(self, model_id):
        """
        Signal that is evoked when a model is done training.

        Starts another instance if the trainer has determined that it is not
        done training
        :param model_id:
        :return:
        """
        if model_id not in self.training_model_ids:
            raise ValueError("Unexpected model id")
        else:
            self.training_model_ids.remove(model_id)
            if not self.is_still_searching():
                self.start_new_instance(model_id)
    def stop_all(self):
        """
        The kill switch. Stops all parallelized trainers
        """
        for model_id in self.training_model_ids:
            # TODO kill the model somehow
            raise NotImplementedError()

    def start(self):
        """
        Starts the training and optimization process
        """
        while len(self.training_model_ids) < self.max_instances:
            hyperparameters = self.hyper_queue.pop()
            self.start_new_instance(hyperparameters)

        while not self.is_still_searching():
            hyperparameters = self.hyper_queue.pop()
            self.start_new_instance(hyperparameters)
