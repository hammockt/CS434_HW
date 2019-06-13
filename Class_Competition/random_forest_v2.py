"""
module for ensemble learning
"""

import random
from model import Model
from decision_tree import DecisionTree

class RandomForest(Model):
    def __init__(self):
        self.trees = []
        self.weights = []

    def add_tree(self, tree):
        self.trees.append(tree)

    def remove_trees(self):
        self.trees = []

    def train(self, row_pointers, x_vals, y_vals, **train_args):
        num_trees = train_args.get("num_trees")
        random_features = train_args.get("random_features")

        num_rows = len(x_vals)
        for i in range(num_trees):
            #random.choices was added in python 3.6!!!
            row_pointers = random.choices(range(num_rows), k=num_rows)

            tree = DecisionTree()
            tree.train(row_pointers, x_vals, y_vals, random_features=random_features)
            self.trees.append(tree)

    def predict(self, point):
        predictions = {}
        for i, tree in enumerate(self.trees):
            prediction = tree.predict(point)

            key = repr(prediction)
            if key not in predictions:
                predictions[key] = 0
            predictions[key] += self.weights[i]

        return float(max(predictions, key=lambda key: predictions[key]))

    def predict_with_confidence(self, point):
        predictions = {}
        for i, tree in enumerate(self.trees):
            prediction = tree.predict(point)

            key = repr(prediction)
            if key not in predictions:
                predictions[key] = 0
            predictions[key] += self.weights[i]

        prediction = float(max(predictions, key=lambda key: predictions[key]))
        confidence = predictions[repr(prediction)] / sum(self.weights)

        return prediction, confidence
