from model import Model
from gensim.models.doc2vec import Doc2Vec
import os
from collections import namedtuple
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import json

def hyperdefault(name, default, hyper_dict):
    """
    Handles the default assignment of parameters that may or may
    not appear in the hyperparameter dict. Adds them to the hyperparameter dict
    """
    if name not in hyper_dict:
        hyper_dict[name] = default
    return hyper_dict[name]
def make_doc_tuple(personA, i):
    analyzedDocument = namedtuple("AnalyzedDocument", "words tags")
    words = personA.lower().split()
    tags = [i]
    return analyzedDocument(words, tags)
def prepare_convpairs(conv_gen, start_i=0):
    """
    Returns a generator that creates a doc tuple
    :param conv_gen:
    :param start_i:
    :return:
    """
    for i, text in enumerate(conv_gen):
        yield make_doc_tuple(text, i + start_i)
def get_nearpy_id(response):
    # response = json.dumps(response)
    return response['id']
class NNModel(Model):
    def __init__(self, sess, source, store_sentences=True, hyperparameters={}, save_dir='./run/'):
        """
        :param sess: tensorflow session to be passed in
        :param conv_pairs: Iterable of conversation pairs
        :param store_sentences: Boolean; whether to store ids or sentences in embedding.
            Choice is dependent on how much storage space you're willing to allocate
            for the nearest neighbors call - sentences are bigger

        :param hyperparameters:
        :param save_dir:
        """
        super().__init__(sess, hyperparameters, save_dir)
        self.store_sentences = store_sentences
        self.source = source
        self.save_file = os.path.join(self.save_dir, 'doc2vec.bin')
        self.construct()
    def setup_doc2vec(self):
        """
        Sets up the doc2vec embedding using the conv_pairs and the set hyperparameters
        :param conv_pairs: A list of tuples corresponding to a conversation pair
        :param hyperparameters: Hyperparameters of the model
        """
        self.dimension = hyperdefault("dimension", 100, self.hyperparameters)
        self.window = hyperdefault("window", 300, self.hyperparameters)
        self.min_count = hyperdefault("min_count", 1, self.hyperparameters)
        self.workers = hyperdefault("workers", 4, self.hyperparameters)
        self.vecs = None
        self.doc2vec = Doc2Vec(size=self.dimension, window=self.window, min_count=self.min_count,
                               workers=self.workers)

    def setup_nn(self):
        """ Sets up nearest neighbors """
        # TODO add bits of entropy as hyperparameters
        print("Setup NN")
        rbp = RandomBinaryProjections('rbp', 9)
        self.nn_engine = Engine(self.dimension, lshashes=[rbp])
        for _, idx in self.source.get_batch():
            vec = self._get_vector(idx)
            self.nn_engine.store_vector(vec, {'id': idx})

    def _get_neighbors(self, vec):
        return self.nn_engine.neighbours(vec)
    def _make_new_vectors(self, string_list):
        """ Takes in a list of strings (NOT TUPLES) and returns their vectors"""
        vecs = []
        for s in string_list:
            vecs.append(self._make_vector(s))
        return vecs
    def _get_vector(self, idx):
        return self.doc2vec.docvecs[idx]
    def _make_vector(self, s):
        return self.doc2vec.infer_vector(s)
    def feed(self, batch_features, batch_labels):
        super().feed(batch_features, batch_labels)
        # return
    def is_trained(self):
        """ Returns whether the model has been trained yet"""
        return self.vecs is not None
    def check_trained(self):
        """ Verifies that the model has been trained or not and throws an error if not"""
        if not self.is_trained():
            raise RuntimeError("You need to train the model first")
    def get_conv_pair(self, id):
        """ Returns the conversation pair tupe corresponding to ID"""
        self.check_trained()
        return self.source.get_convpair(id)
    def get_data_response(self, id):
        """ Returns the response to the conversation """
        return self.source.get_response(id)
    def get_response(self, prev): 
        self.nn_engine.fetch_vector_filters = None
        vec = self._make_vector(prev)
        neighbors = self._get_neighbors(vec)
        neighbors = sorted(neighbors, key=lambda x: x[2])
        print(neighbors)
        if neighbors:
            res =  get_nearpy_id(neighbors[0][1])
        else:
            res = "588a865fc280444b37064f11"
        return self.get_data_response(res)

    def train(self):
        super().train_batch(None, None)

        processed_convs = self.get_convpairs()
        #self.doc2vec = Doc2Vec(documents=processed_convs, size=self.dimension, window=self.window, min_count=self.min_count,
        #                        workers=self.workers)
        self.doc2vec.build_vocab(processed_convs, trim_rule=None)
        self.doc2vec.train(processed_convs)
        self.vecs = self.doc2vec.docvecs
        print(self.vecs)
        self.setup_nn()

    def get_convpairs(self):
        for text, id in self.source.get_batch():
            yield make_doc_tuple(text, id)
    def close(self):
        super().close()

    def save(self):
        """ Saves the doc2vec embedding """
        super().save()
        self.doc2vec.save(self.save_file)


    def load(self):
        super().load()

        self.doc2vec = Doc2Vec.load(self.save_file)
        self.setup_nn()

    def construct(self):
        super().construct()
        self.setup_doc2vec()
        # TODO test later
        # if os.path.isfile(self.save_file):
        #     self.load()

    def test(self, batch_features, batch_labels):
        super().test(batch_features, batch_labels)
