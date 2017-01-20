class ModelStore:
    """
    The class that handles the model storage system.
    Can take two forms
    1: the local file handler - serves models using the local fs to clients that
    request it
    2: the remote server handler - serves files by querying a running ModelStore server
    on another machine
    """
    def __init__(self, path_to_fs, remote=False):
        """
        Setups the model store
        :param path_to_fs: The path to the store.
        Directory if remote=False
        IP if remote=True
        :param remote: flag whether or not this is a remote model store
        """
        raise NotImplementedError()
        self.remote = remote
        if remote:
            # TODO check whether path_to_fs is an ip
            # TODO check whether the remote model_store is running
            pass
        else:
            # TODO check whether path_to_fs is a directory
            pass

    def list_ids(self):
        """
        Prints out a list of model_ids stored in the filestore
        :return:
        """
        # TODO implement
        raise NotImplementedError()

    def has_model(self, id):
        """
        Returns whether a model with the id is in the fs
        :param id: The id of the model to check
        :return:
        """
        # TODO implement
        raise NotImplementedError()

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
