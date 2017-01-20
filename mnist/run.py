import tensorflow as tf
from model_store import ModelStore
from mnist_evaluator import MNISTEvaluator
from mnist_softmax import MNISTSoftmax
from mnist_trainer import MNISTTrainer

def main(_):
    print('Loading Model Store')
    store = ModelStore(path_to_fs='run/')

    print('Model Ids :', store.list_ids())
    evaluator = MNISTEvaluator(model_store=store)

    print("Setting up Training")
    trainer = MNISTTrainer(model=MNISTSoftmax,
                      evaluator=evaluator,
                      model_=store)

    print("Start Training")
    trainer.start(verbose=True)


if __name__ == '__main__':
    tf.app.run()
