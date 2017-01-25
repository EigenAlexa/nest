import tensorflow as tf
from model_store import ModelStore
from mnist_evaluator import MNISTEvaluator
from mnist_softmax import MNISTSoftmax
from mnist_trainer import MNISTTrainer
from mnist_hyperqueue import MNISTHyperQueue
from mnist_datasource import MNISTDataSource

def main(_):
    print('Loading Model Store')
    store = ModelStore(path_to_fs='run/')

    print('Model Ids :', store.list_ids())
    evaluator = MNISTEvaluator(model_store=store)
    hyper_queue = MNISTHyperQueue()
    data_source = MNISTDataSource()
    print("Setting up Training")
    trainer = MNISTTrainer(model_class=MNISTSoftmax,
                      evaluator=evaluator,
                      data_source=data_source,
                      model_store=store,

                      hyper_queue=hyper_queue)

    print("Start Training")
    trainer.start(verbose=True)


if __name__ == '__main__':
    tf.app.run()
