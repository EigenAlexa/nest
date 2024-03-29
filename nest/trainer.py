import tensorflow as tf

class Trainer:
    """
    The class that handles training of a specific model.
    Handle hyperparameter search, model instance distribution,
    and setting up the evaluation scheme
    """

    def __init__(self, model_class, data_source, evaluator, model_store, hyper_queue, max_instances=1):
        self.model_class = model_class
        self.evaluator = evaluator
        self.data_source = data_source
        self.model_store = model_store
        self.hyper_queue = hyper_queue
        self.training_model_ids = set()
        self.max_instances = max_instances

    def start_new_instance(self, hyperparameters):
        """
        Takes in a set of hyperparameters, initializes  and starts training a new instance of model_class
        :return: model_id for new instance
        """
        with tf.Session() as sess:
            self.cur_model = self.model_class(sess, hyperparameters)
            id = self.cur_model.model_id
            assert id not in self.training_model_ids, "Duplicate id found, aborting"
            self.training_model_ids.add(id)
            tf.global_variables_initializer().run()
            if "num_epochs" not in hyperparameters:
                num_epochs = 10
            else:
                num_epochs = hyperparameters['num_epochs']
            if "batch_size" not in hyperparameters:
                batch_size = 64
            else:
                batch_size = hyperparameters['batch_size']

            for epoch in range(num_epochs):
                for batch in range(self.data_source.get_num_batches(batch_size)):
                    batchx, batchy = self.data_source.get_batch(batch_size)
                    self.cur_model.train_batch(batchx, batchy)
            self.done_training(id) 
        # TODO Allocate the resources for the model
        # TODO Create the model with the set of hyperparameters
        # TODO Setup the datasource with the model
        # TODO Start training
        # TODO provide some sort of callback that will trigger self.done_training()

    def continue_training_instance(self, model_id):
        """
        Loads in a pretrained model from model_id and continues training """
        # TODO Allocate the resources for the model
        # TODO Load the model resources from the model_store
        # TODO Load the model from the model_spec
        # TODO Setup the datasource with the model
        # TODO Continue training
        # TODO provide some sort of callback that will trigger self.done_training()
        raise NotImplementedError()
    def evaluate_instance(self, model_id):
        """ Evaluates a model instance. """
        # Takes in the model, grabs the id and passes that to the evaluator.
        # The evaluator then pulls the model files from the model server,
        # reconstructs the graph, loads the checkpoint, and
        # TODO implement

        raise NotImplementedError()

    def is_still_searching(self):
        """
        :return: Boolean whether the training method should stop
        """
        # TODO figure out what else should be done here
        return not self.hyper_queue.has_more()

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
            # TODO add the model to the model_store
            self.cur_model.save()
            self.model_store.save_current_model(model_id, self.cur_model.save_dir)
            self.training_model_ids.remove(model_id)
            if not self.is_still_searching():
                self.start_new_instance(self.hyper_queue.pop())
    def stop_training(self, model_id):
        """
        Tells a model with the id to stop training
        :param model_id:
        :return:
        """
        # handle the case whether a model id is not in the trainer
        if model_id not in self.training_model_ids:
            raise ValueError("Model with id {} not contained by this trainer".format(model_id))
        raise NotImplementedError()
    def stop_all(self):
        """
        The kill switch. Stops all parallelized trainers
        """
        for model_id in self.training_model_ids:
            self.stop_training(model_id)

    def start(self, verbose=False):
        """
        Starts the training and optimization process
        """
        while len(self.training_model_ids) < self.max_instances:
            hyperparameters = self.hyper_queue.pop()
            self.start_new_instance(hyperparameters)
