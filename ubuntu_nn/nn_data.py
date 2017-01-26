from datasource import DataSource
from pymongo import MongoClient
import bson

class NNData(DataSource):
    def __init__(self, source='10.0.2.32', is_local=False):
        """ source is mongo db ip"""
        super().__init__(source, is_local)
        self.client = MongoClient(source)
        self.gen = self.client['ubuntu-corpus'].dialogs.find()

    def get_batch(self, batch_size=None):
        super().get_batch(batch_size)
        for next_elm in self.gen:
            personA = next_elm['A']
            id = next_elm['_id']

            yield personA, str(id)

    def get_response(self, idx):
        """ Returns the response from mongo db at index"""
        next_elm = self.get_convpair(idx)
        personB = next_elm['B']
        return personB
    def get_convpair(self, idx):
        return self.client['ubuntu-corpus'].dialogs.find({'_id' : bson.objectid.ObjectId(idx)}).next()
    def get_ids(self):