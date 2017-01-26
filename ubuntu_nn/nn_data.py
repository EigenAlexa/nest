from datasource import DataSource
from pymongo import MongoClient

def generator(iter):
    for x in iter:
        yield x
def get_id
class NNData(DataSource):
    def __init__(self, source='10.0.2.32', is_local=False):
        """ source is mongo db ip"""
        super().__init__(source, is_local)
        self.client = MongoClient(source)
        self.gen = self.client['ubuntu-corpus'].dialogs.find()

    def get_batch(self, batch_size):
        super().get_batch(batch_size)
        next_elm = self.gen.next()
        personA = next_elm['A']
        id = next_elm['_id']
        return personA, id

    def get_response(self, idx):
        """ Returns the response from mongo db at index"""
        next_elm = self.client['ubuntu-corpus'].dialogs.find({'_id' : idx}).next()
        personB = next_elm['B']
        return personB
    def get_id_gen(self):