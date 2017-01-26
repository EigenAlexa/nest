from model import Model
from gensim.models.doc2vec import Doc2Vec
import os
from collections import namedtuple
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

def hyperdefault(name, default, hyper_dict):
    """
    Handles the default assignment of parameters that may or may
    not appear in the hyperparameter dict. Adds them to the hyperparameter dict
    """
    if name not in hyper_dict:
        hyper_dict[name] = default
    return hyper_dict[name]
def make_doc_tuple(conv_pair, i):
    analyzedDocument = namedtuple("AnalyzedDocument", "words tags")
    words = conv_pair[0].lower().split()
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
class NNModel(Model):
    def __init__(self, sess, conv_pairs, store_sentences=True, hyperparameters={}, save_dir='./run/'):
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
        self.conv_pairs = conv_pairs
        self.store_sentences = store_sentences
        self.construct()
    def setup_doc2vec(self):
        """
        Sets up the doc2vec embedding using the conv_pairs and the set hyperparameters
        :param conv_pairs: A list of tuples corresponding to a conversation pair
        :param hyperparameters: Hyperparameters of the model
        """
        self.dimension = hyperdefault("dimension", 100, self.hyperparameters)
        window = hyperdefault("window", 300, self.hyperparameters)
        min_count = hyperdefault("min_count", 1, self.hyperparameters)
        workers = hyperdefault("workers", 4, self.hyperparameters)
        processed_convs = prepare_convpairs(self.conv_pairs)
        self.doc2vec = Doc2Vec(documents=processed_convs, size=self.dimension, window=window, min_count=min_count,
                               workers=workers)
        self.vecs = self.doc2vec.docvecs
    def setup_nn(self):
        """ Sets up nearest neighbors """
        # TODO add bits of entropy as hyperparameters
        rbp = RandomBinaryProjections('rbp', 10)
        self.nn_engine = Engine(self.dimension, lshashes=[rbp])

        for i, vec in enumerate(self.vecs):
            if self.store_sentences:
                self.nn_engine.store_vector(vec, self.get_train_response(i))
            else:
                self.nn_engine.store_vector(vec, i)

    def _get_neighbors(self, vec):
        return self.nn_engine.neighbours(vec)
    def _make_new_vectors(self, string_list):
        """ Takes in a list of strings (NOT TUPLES) and returns their vectors"""
        vecs = []
        for s in string_list:
            vecs.append(self._make_vector(s))
        return vecs
    def _make_vector(self, s):
        return self.doc2vec.infer_vector(s)
    def feed(self, batch_features, batch_labels):
        super().feed(batch_features, batch_labels)
        # return
    def get_conv_pair(self, id):
        """ Returns the conversation pair tupe corresponding to ID"""
        return self.conv_pairs[id]
    def get_train_response(self, id):
        """ Returns the response to the conversation """
        return self.get_conv_pair(id)[1]
    def get_response(self, prev):
        vec = self._make_vector(prev)
        neighbors = self._get_neighbors(vec)
        # TODO figure out if this is the right format
        return neighbors#[0][1]


    def train_batch(self, X, y):
        super().train_batch(X, y)
        # Doesn't really do anything because training should be done initially
        pass

    def close(self):
        super().close()

    def save(self):
        """ Saves the doc2vec embedding """
        super().save()
        self.doc2vec.save(os.path.join(self.save_dir,'doc2vec.bin'))


    def load(self):
        super().load()

    def construct(self):
        super().construct()
        self.setup_doc2vec()
        self.setup_nn()

    def test(self, batch_features, batch_labels):
        super().test(batch_features, batch_labels)