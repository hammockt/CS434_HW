"""
Interface that all of our models will implement
"""

from abc import abstractmethod
import json

class Model():
    """ represents a model or classifier in ML """

    @abstractmethod
    def train(self, row_pointers, x_vals, y_vals, **train_args):
        """ train the model """
        raise NotImplementedError
    @abstractmethod
    def predict(self, point):
        """ predict the class label for the given point """
        raise NotImplementedError

    def to_json(self):
        """ helper to turn a model into json string, irregardless if it is recursive """
        return json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)))
