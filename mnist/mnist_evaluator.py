from evaluator import Evaluator
from model import DataSetType
from mnist_data import MNISTData

class MNISTEvaluator(Evaluator):

    def __init__(self, model_store):
        self.model_store = model_store
        self.data_source = MNISTData()
        self.data_source.set_dataset_type(DataSetType.TEST)

    def evaluate(self, model):
        # Test
        # TODO check this out
        # this seems kind of a waste of opportunity to actually have that here
        # maybe it's just the simplicity of the model and the evaluation scheme.
        return model.test(*self.data_source.get_batch(1000))
        # acc = 0
        # for i in range(1000):
        #     batch_xs, batch_ys = self.data_source.get_batch(100)
        #
        #     if i % 50 == 0:
        #         self.data_source.set_dataset_type(DataSetType.TEST)
        #         acc = model.test(*self.data_source.get_batch(1000))
        #         print('Iteration {}; Accuracy {}'.format(i, acc))
        #         self.data_source.set_dataset_type(DataSetType.TRAIN)
        #     else:
        #         model.feed(batch_xs, batch_ys, i, execution_stats=(i % 100 == 99))
        # return acc
