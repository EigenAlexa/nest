from evaluator import Evaluator
from model import DataSetType
from mnist_datasource import MNISTDataSource

class MNISTEvaluator(Evaluator):

    def __init__(self, model_store):
        self.model_store = model_store
        self.data_source = MNISTDataSource()
        self.data_source.set_dataset_type(DataSetType.TEST)

    def evaluate(self, model):
        # Test
        # TODO check this out
        # this seems kind of a waste of opportunity to actually have that here
        # maybe it's just the simplicity of the model and the evaluation scheme.
        return model.test(*self.data_source.get_batch(1000))

