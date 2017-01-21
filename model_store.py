import os

class ModelStore:
    """
    The class that handles the model storage system.
    Can take two forms
    1: the local file handler - serves models using the local fs to clients that
    request it
    2: the remote server handler - serves files by querying a running ModelStore server
    on another machine

    For the local machine, the folders in the directory *must be strictly* ids
    """
    def __init__(self, path_to_fs, remote=False):
        """
        Setups the model store
        :param path_to_fs: The path to the store.
            A Directory if remote=False
            IP if remote=True
        :param remote: flag whether or not this is a remote model store
        """
        self.remote = remote
        if self.remote:
            # TODO check whether path_to_fs is an ip
            # TODO check whether the remote model_store is running
            raise NotImplementedError()
        else:
            # check whether path_to_fs is a directory and raise error if not
            if not os.path.isdir(path_to_fs):
                raise ValueError("Directory specified was not found")
            self.path_to_fs = path_to_fs

    def list_ids(self):
        """
        Prints out a list of model_ids stored in the filestore
        :return:
        """
        # TODO implement
        if self.remote:
            raise NotImplementedError()
        else:
            # TODO read the directory
            ids = [f.path for f in os.scandir(self.path_to_fs) if f.is_dir()]
            return ids


    def has_model(self, id):
        """
        Returns whether a model with the id is in the fs
        :param id: The id of the model to check
        :return:
        """
        return id in self.list_ids()

    def get_model(self, id):
        """
        Returns a model that corresponds to an id; otherwise returns false
        :param id: The id of the model to check
        :return: Model reference
        """
        if self.has_model(id):
            raise NotImplementedError()
        else:
            raise FileNotFoundError("Model with id {} could not be found".format(id))

    def add_model(self, model):
        """
        Takes in a trained model, saves the model, and moves summaries,
        checkpoints, and spec to the model file_store.
        """
        raise NotImplementedError()
