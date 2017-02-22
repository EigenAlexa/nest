from nest.hyperqueue import HyperQueue

class MNISTHyperQueue(HyperQueue):
    """
    HyperQueue for MNIST Softmax regression.
    Simply contains a list as the backend
    with different learning rate values in hyperparam
    dicts.
    """
    def __init__(self):
        super().__init__()
        self._queue = [{'lr' : lr} for lr in [0.1, 0.01, 0.001]]
    def pop(self):
        super().pop()
        elm = self._queue.pop()
        print("Popping MNIST HypyrQueue", elm)
        return elm

    def has_more(self):
        return len(self._queue) > 0

    def update_priority_fn(self, priority_fn):
        """ Does nothing because this is a simple test"""
        super().update_priority_fn(priority_fn)

    def push(self, hyperparams):
        super().push(hyperparams)
        if type(hyperparams) == list:
            self._queue.extend(hyperparams)
        else:
            self._queue.append(hyperparams)
